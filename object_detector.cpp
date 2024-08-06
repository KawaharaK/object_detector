#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

namespace {

    const int MIN_AREA = 20;  // 最小面積（ピクセル数）
    const int MAX_AREA = 10000;  // 最大面積（ピクセル数）
    const double MIN_CIRCULARITY = 0.5;  // 最小真円度
    const double PIXEL_TO_LENGTH = 0.5;  // 例: 1ピクセルが0.5μmの場合
    const char* WINDOW_NAME = "Detected Objects";
    const char* INPUT_IMAGE_PATH = "sample.png";
    const char* OUTPUT_CSV_PATH = "object_features.csv";

    double calculateCircularity(const std::vector<cv::Point>& contour) {
        double area = cv::contourArea(contour);
        double perimeter = cv::arcLength(contour, true);
        return (perimeter > 0) ? 4 * CV_PI * area / (perimeter * perimeter) : 0;
    }

    void saveToCSV(const std::string& filename, const std::vector<std::vector<double>>& data) {
        std::ofstream outputFile(filename);
        if (!outputFile.is_open()) {
            throw std::runtime_error("ファイルを開けませんでした。");
        }

        outputFile << "Label,CentroidX,CentroidY,Area,Circularity\n";
        for (const auto& row : data) {
            for (size_t i = 0; i < row.size(); ++i) {
                outputFile << row[i];
                if (i < row.size() - 1) outputFile << ",";
            }
            outputFile << "\n";
        }
    }

    cv::Mat preprocessImage(const cv::Mat& input) {
        cv::Mat denoised, binary, morphed;
        cv::GaussianBlur(input, denoised, cv::Size(5, 5), 0);
        cv::threshold(denoised, binary, 160, 255, cv::THRESH_BINARY);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(binary, morphed, cv::MORPH_OPEN, kernel);
        return morphed;
    }

} // anonymous namespace

int main() {
    try {
        // OpenCVのログレベルをERRORに設定し、不要な情報出力を抑制
        cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);


        // 入力画像をグレースケールで読み込み
        cv::Mat img = cv::imread(INPUT_IMAGE_PATH, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            throw std::runtime_error("画像を読み込めませんでした。");
        }

        // 読み込んだ画像の寸法とチャンネル数を表示
        std::cout << "画像のサイズ: " << img.size().width << " x " << img.size().height << " x " << img.channels() << std::endl;

        // 画像の前処理（ノイズ除去、2値化、モルフォロジー処理）を実行
        cv::Mat preprocessed = preprocessImage(img);

        // 前処理された画像から輪郭を検出
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(preprocessed, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 連結成分のラベリングを実行し、各オブジェクトの統計情報を取得
        cv::Mat labels, stats, centroids;
        int nLabels = cv::connectedComponentsWithStats(preprocessed, labels, stats, centroids, 8, CV_32S);

        // 輪郭とラベルを対応付けるマップを作成
        std::map<int, int> labelToContourMap;

        for (int i = 0; i < contours.size(); ++i) {
            // 各輪郭の中心点を計算
            cv::Moments m = cv::moments(contours[i]);
            int cx = static_cast<int>(m.m10 / m.m00);
            int cy = static_cast<int>(m.m01 / m.m00);

            // 中心点に対応するラベルを取得
            int label = labels.at<int>(cy, cx);

            // マップに追加
            labelToContourMap[label] = i;
        }

        // 検出されたオブジェクトの数を表示（背景ラベルを除く）
        std::cout << "検出されたオブジェクトの数: " << nLabels - 1 << std::endl;  

        // ラベリング結果の可視化のための準備
        // 色付けされたラベル画像と、フィルタリングされたオブジェクトを描画するための画像を作成
        cv::Mat coloredLabels, filteredImage = cv::Mat::zeros(img.size(), CV_8UC3);

        // ラベル画像を8ビット単チャンネル画像に変換
        // 各ラベルの値を0-255の範囲に正規化
        labels.convertTo(coloredLabels, CV_8U, 255.0 / (nLabels - 1));

        // 正規化されたラベル画像にカラーマップを適用
        // COLORMAP_JETは青から赤までの連続的な色変化を持つ
        cv::applyColorMap(coloredLabels, coloredLabels, cv::COLORMAP_JET);

        // フィルタリングされたオブジェクトのデータを格納するベクトル
        std::vector<std::vector<double>> csvData;

        // 新しいラベル番号の初期化
        int newLabelNo = 1;  

        // 各オブジェクトに対して処理を行う
        for (int i = 1; i < nLabels; i++) {
            // オブジェクトの特徴量を取得
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            double centerX = centroids.at<double>(i, 0);
            double centerY = centroids.at<double>(i, 1);

            // 対応する輪郭のインデックスを取得
            int contourIndex = labelToContourMap[i];

            // 真円度を計算
            double circularity = calculateCircularity(contours[contourIndex]);

            // フィルタリング条件を満たすオブジェクトのみ処理
            if (area >= MIN_AREA && area <= MAX_AREA && circularity >= MIN_CIRCULARITY) {
                // CSVデータ用に面積をμm単位に変換
                double convertedArea = area * PIXEL_TO_LENGTH * PIXEL_TO_LENGTH;

                // フィルタリングされたオブジェクトのデータをCSV用に保存
                csvData.push_back({ static_cast<double>(newLabelNo), centerX, centerY, static_cast<double>(convertedArea), circularity });

                // フィルタリングされたオブジェクトの輪郭を描画
                cv::drawContours(filteredImage, contours, contourIndex, cv::Scalar(0, 255, 0), 2);

                // オブジェクトの中心に新しいラベル番号を描画
                cv::putText(filteredImage, std::to_string(newLabelNo), cv::Point(centerX, centerY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);  


                // 次のラベル番号に進む
                newLabelNo++;
            }
        }

        // フィルタリングされたオブジェクトのデータをCSVファイルに保存
        saveToCSV(OUTPUT_CSV_PATH, csvData);

        // 画像の表示
        //cv::imshow(WINDOW_NAME, preprocessed);
        cv::imshow(WINDOW_NAME, filteredImage);
        cv::waitKey(0);

    }
    catch (const std::exception& e) {
        // エラーメッセージを標準エラー出力に表示
        std::cerr << "エラー: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
