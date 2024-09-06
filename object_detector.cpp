#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <filesystem> 

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

    // プログラムの設定パラメータを保持する構造体
    struct ProgramParameters {
        int minArea;
        int maxArea;
        double minCircularity;
        double pixelToLength;
        std::string inputImagePath;
        std::string outputCsvPath;
    };

    // 設定ファイルからパラメータを読み込む関数
    ProgramParameters loadConfig(const std::string& configPath) {
        // 設定ファイルを開く
        std::ifstream configFile(configPath);

        // ファイルが正常に開けなかった場合、例外をスロー
        if (!configFile.is_open()) {
            throw std::runtime_error("設定ファイルを開けませんでした: " + configPath);
        }

        // JSONオブジェクトを作成し、ファイルの内容を解析
        json config;
        configFile >> config;

        // ProgramParameters構造体のインスタンスを作成
        ProgramParameters params;

        // JSONオブジェクトから各パラメータを読み取り、構造体に格納
        params.minArea = config["minArea"];
        params.maxArea = config["maxArea"];
        params.minCircularity = config["minCircularity"];
        params.pixelToLength = config["pixelToLength"];
        params.inputImagePath = config["inputImagePath"];
        params.outputCsvPath = config["outputCsvPath"];

        return params;
    }

    // 輪郭の真円度を計算する関数
    double calculateCircularity(const std::vector<cv::Point>& contour) {
        double area = cv::contourArea(contour);
        double perimeter = cv::arcLength(contour, true);
        return (perimeter > 0) ? 4 * CV_PI * area / (perimeter * perimeter) : 0;
    }

    // 文字列を小文字に変換する関数
    std::string toLower(const std::string& s) {
        // 入力文字列のコピーを作成
        std::string result = s;

        // 文字列内の各文字を小文字に変換
        std::transform(result.begin(), result.end(), result.begin(),
            [](unsigned char c) { return std::tolower(c); });
        return result;
    }

    // ファイルが画像ファイルかどうかを判定する関数
    bool isImageFile(const fs::path& filePath) {
        std::string ext = toLower(filePath.extension().string());
        return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tif";
    }

    // フォルダ内の画像ファイルのリストを取得する関数
    std::vector<fs::path> getImageFiles(const fs::path& folderPath) {
        // 画像ファイルのパスを格納するためのベクター
        std::vector<fs::path> imageFiles;

        // フォルダ内の各エントリ（ファイルやサブディレクトリ）をイテレート
        for (const auto& entry : fs::directory_iterator(folderPath)) {
            // エントリが通常のファイルであり、かつ画像ファイルであるかチェック
            if (fs::is_regular_file(entry) && isImageFile(entry.path())) {
                // 条件を満たす場合、そのファイルのパスをベクターに追加
                imageFiles.push_back(entry.path());
            }
        }
        return imageFiles;
    }

    // ファイルパスからディレクトリパスを取得する関数
    fs::path getDirectoryPath(const fs::path& filePath) {
        return filePath.parent_path();
    }

    // ファイルパスからファイル名を取得する関数
    std::string getFileName(const fs::path& filePath) {
        return filePath.filename().string();
    }

    // ファイル名から拡張子を取得する関数
    std::string getFileExtension(const fs::path& filePath) {
        return filePath.extension().string();
    }

    // ファイル名から拡張子を除いた部分を取得する関数
    std::string getFileNameWithoutExtension(const fs::path& filePath) {
        return filePath.stem().string();
    }

    // 検出結果を描画した画像を保存する関数
    void saveDetectedImage(const cv::Mat& detectedImage, const fs::path& inputImagePath) {
        // 出力ファイルのパスを生成
        // 1. 元の画像と同じディレクトリに保存
        // 2. ファイル名の末尾に "_detected" を追加
        // 3. 元の画像と同じ拡張子を使用
        fs::path outputPath = inputImagePath.parent_path() / (inputImagePath.stem().string() + "_detected" + inputImagePath.extension().string());

        // OpenCVのimwrite関数を使用して画像を保存
        // outputPath.string()でパスを文字列に変換
        cv::imwrite(outputPath.string(), detectedImage);

        // 保存完了メッセージをコンソールに出力
        std::cout << "検出結果を保存しました: " << outputPath << std::endl;
    }

    // CSVファイルにデータを保存または追記する関数
    void saveOrAppendToCSV(const fs::path& imagePath, const std::string& outputCsvPath, const std::vector<std::vector<double>>& data, bool isFirstImage) {
        // 画像が存在するフォルダのパスを取得
        fs::path imageFolder = imagePath.parent_path();

        // 画像フォルダパスとCSVファイル名を結合
        fs::path fullCsvPath = imageFolder / outputCsvPath;

        // ファイルを開くモードを設定（最初の画像なら上書き、それ以外なら追記）
        std::ios_base::openmode mode = isFirstImage ? std::ios::out : std::ios::app;

        // ファイルを開く
        std::ofstream outputFile(fullCsvPath, mode);

        // ファイルが正常に開けなかった場合、例外をスロー
        if (!outputFile.is_open()) {
            throw std::runtime_error("ファイルを開けませんでした: " + fullCsvPath.string());
        }

        // 最初の画像の場合、ヘッダーを書き込む
        if (isFirstImage) {
            outputFile << "ImageName,Label,Centroid X,Centroid Y,Area,Effective diameter,Circularity\n";
        }

        // 現在処理中の画像ファイル名を取得
        std::string imageName = imagePath.filename().string();

        // データを CSV 形式で書き込む
        for (const auto& row : data) {
            // 各行の先頭に画像ファイル名を追加
            outputFile << imageName;

            // 行内の各値をカンマ区切りで追加
            for (const auto& value : row) {
                outputFile << "," << value;
            }

            // 行末に改行を追加
            outputFile << "\n";
        }

        // 処理成功メッセージをコンソールに出力
        std::cout << (isFirstImage ? "CSVファイルを作成しました: " : "CSVファイルに追記しました: ") << fullCsvPath << std::endl;
    }

    // 画像の前処理を行う関数
    cv::Mat preprocessImage(const cv::Mat& input) {
        cv::Mat denoised, binary, morphed;

        // ガウシアンブラーを適用してノイズを軽減
        cv::GaussianBlur(input, denoised, cv::Size(7, 7), 0);

        // 二値化処理
        cv::threshold(denoised, binary, 50, 255, cv::THRESH_BINARY_INV);

        // モルフォロジー処理（オープニングとクロージング）
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));

        // 小さなノイズの除去と、オブジェクトの分離に有効
        cv::morphologyEx(binary, morphed, cv::MORPH_OPEN, kernel);

        // 小さな穴や隙間を埋める効果がある
        cv::morphologyEx(morphed, morphed, cv::MORPH_CLOSE, kernel);

        return morphed;
    }

} // anonymous namespace


int main(int argc, char* argv[]) {
    try {
        // コマンドライン引数のチェック
        if (argc != 2) {
            throw std::runtime_error("Usage: " + std::string(argv[0]) + " <config_file_path>");
        }

        // 設定ファイルからパラメータを読み込む
        ProgramParameters params = loadConfig(argv[1]);
        fs::path folderPath = params.inputImagePath;

        // フォルダ内の画像ファイル情報を取得
        std::vector<fs::path> imageFiles = getImageFiles(folderPath);

        // OpenCVのログレベルをERRORに設定し、不要な情報出力を抑制
        cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

        // 各画像に対して処理を行うループ
        bool isFirstImage = true;
        for (const auto& imagePath : imageFiles) {
            std::cout << "\n処理中の画像: " << imagePath << std::endl;

            // 入力画像をグレースケールで読み込み
            cv::Mat img = cv::imread(imagePath.string(), cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                std::cerr << "警告: 画像を読み込めませんでした: " << imagePath << std::endl;
                continue;  // 次の画像に進む
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

            // ラベリング結果の可視化のための準備
            cv::Mat coloredLabels, filteredImage = cv::Mat::zeros(img.size(), CV_8UC3);

            // ラベル画像を8ビット単チャンネル画像に変換し、各ラベルの値を0-255の範囲に正規化
            labels.convertTo(coloredLabels, CV_8U, 255.0 / (nLabels - 1));

            // 正規化されたラベル画像にカラーマップを適用
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
                if (area >= params.minArea && area <= params.maxArea && circularity >= params.minCircularity) {
                    // 直径と有効半径を計算
                    double effectiveDiameter = 2 * std::sqrt(area / CV_PI) * params.pixelToLength;
                    
                    // CSVデータ用に面積をμm単位に変換
                    double convertedArea = area * params.pixelToLength * params.pixelToLength;

                    // フィルタリングされたオブジェクトのデータをCSV用に保存
                    csvData.push_back({ static_cast<double>(newLabelNo), centerX, centerY, static_cast<double>(convertedArea), effectiveDiameter, circularity});

                    // フィルタリングされたオブジェクトの輪郭を描画
                    cv::drawContours(filteredImage, contours, contourIndex, cv::Scalar(0, 255, 0), 2);

                    // オブジェクトの中心に新しいラベル番号を描画
                    cv::putText(filteredImage, std::to_string(newLabelNo), cv::Point(centerX, centerY),
                        cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(255, 255, 255), 1);

                    // 次のラベル番号に進む
                    newLabelNo++;
                }
            }

            // 検出されたオブジェクトの数を表示（背景ラベルを除く）
            std::cout << "検出されたオブジェクトの数: " << newLabelNo + 1 << std::endl;

            // フィルタリングされたオブジェクトのデータをCSVファイルに保存または追記
            saveOrAppendToCSV(imagePath, params.outputCsvPath, csvData, isFirstImage);

            // 検出結果を元の画像名 + "detected" で保存
            saveDetectedImage(filteredImage, imagePath);

            // 最初の画像の処理が終わったらフラグを false に設定
            if (isFirstImage) {
                isFirstImage = false;
            }
        }
    }
    catch (const std::exception& e) {
        // エラーメッセージを標準エラー出力に表示
        std::cerr << "エラー: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
