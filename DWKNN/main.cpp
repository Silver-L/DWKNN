/**
* @Distance-weighted k-nearest Neighbor(DWKNN) for B4 Contest
* @Reference:Gou, J., Du, L., Zhang, Y., & Xiong, T. (2012). A new distance-weighted k-nearest neighbor classifier. J. Inf. Comput. Sci, 9(February), 1429¨C1436.
* @Caution:All of the features need to be the same typename!!!
*
* @Author: Zhihui Lu
* @Date: 2018/06/18
**/

#include<iostream>
#include<string>
#include<vector>
#include<iomanip>
#include"DWKNN.cpp"
#include"dataIO.h"
#include"ItkImageIO.h"
#include"popcnt_JI.h"
#include"sort_index.h"

typedef double T_learn;
typedef double T_test;
typedef unsigned char T_label;

int main(int argc, const char *argv[]) {

	if (argc != 11 && argc != 12) {
		std::cerr << "usage: [.exe] [feature_list_path] [learn_folder_path] [learn_data_name_list_path] "
			<< "[learn_mask_list_path] [learn_tumor_list_path] [test_folder_path] [test_data_name_list_path] "
			<< "[test_mask_list_path] [outdir] ([test_data_tumor])" << std::endl;
		return EXIT_FAILURE;
	}

	const int num_k = std::stoi(argv[1]);
	const std::string feature_list_path = argv[2];
	const std::string learn_folder_path = argv[3];
	const std::string learn_data_name_list_path = argv[4];
	const std::string learn_mask_list_path = argv[5];	//(.raw)
	const std::string learn_tumor_list_path = argv[6];	//(.raw)
	const std::string test_folder_path = argv[7];
	const std::string test_data_name_list_path = argv[8];
	const std::string test_mask_list_path = argv[9];	//(.raw)
	const std::string outdir = argv[10];
	std::string test_tumor_list_path;			//for optimization(.raw)
		
	if (argc == 12) {
		test_tumor_list_path = argv[11];
	}

	const int size_x = 512;					//image size
	const int size_y = 1024;
	const int num_learn = 10;				//number of learning data
	const int num_test = 10;				//number of test data
	const int max_num_tumor = 10;				//max number of tumor
	const int max_num_mask = 70;				//max number of mask

	int num_feature;					//num of feature
	int image_index;					//for saving image
	int element;						//for calculate JI
	double ji_max;

	bool optimize_k = true;					//for k's optimization
	bool WKNN = true;					//DWKNN


	DWKNN dwknn;
	ImageIO<2> imageio;

	imageio.SetSize(0, size_x);
	imageio.SetSize(1, size_y);
	imageio.SetSpacing(0, 2.8);
	imageio.SetSpacing(1, 2.8);

	std::list<std::string> feature_list;					//name list of feature
	std::list<std::string> learn_name_list;					//name list of learn data
	std::list<std::string> test_name_list;					//name list of test data
	std::string learn_txt_dir;						//for saving learning feature list
	std::string test_txt_dir;						//for saving test feature list

	Eigen::Matrix<T_label, Eigen::Dynamic, Eigen::Dynamic> learn_mask;
	Eigen::Matrix<T_label, Eigen::Dynamic, Eigen::Dynamic> learn_tumor;
	Eigen::Matrix<T_label, Eigen::Dynamic, Eigen::Dynamic> test_mask;
	Eigen::Matrix<T_label, Eigen::Dynamic, Eigen::Dynamic> test_tumor;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> learning_sample;

	std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> learn_data;					//learn_data ( size: num_of_data * (mask_pixel * feature))	(dim: 3)
	std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> test_data;					//test_data ( size: num_of_data * (mask_pixel * feature )) (dim: 3)
	std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>> distance;			//test_distance( size: num_of_test_data * (num_pixel * ( num_sample * num_feature)))
	std::vector<std::vector<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>>> test_classified_index;		//test_classified_index( size: num_of_test_data * (num_pixel * ( num_sample * num_feature)))

	std::vector<std::string> each_learn_txt_path;							//path of all features of each learning data(.txt) for loading data
	std::vector<std::string> each_test_txt_path;							//path of all features of each test data(.txt) for loading data
	std::vector<std::vector<int>> learn_mask_index;
	std::vector<std::vector<int>> test_mask_index;
	std::vector<std::vector<int>> class_tumor_index;						//tumor's pixel index
	std::vector<std::vector<int>> class_mask_index;							//mask's pixel index (not include tumor)
	std::vector<std::vector<unsigned char>> result_label;
	std::vector<std::vector<double>> ji_vector;							//for saving csv

	std::vector<unsigned long long*>test_data_packed;						//for calculate JI
	std::vector<double> learn_classified;								//tumor: +1 ,	mask: -1
	std::vector<double>	ji;
	std::vector<double> ji_mean;
	std::vector<int> classified_index;								//for sorting
	std::vector<int> k_vector;									//for saving csv

	num_feature = dataIO::count_number_of_text_lines(feature_list_path);	//get num of feature

	/* create feature path */
	dataIO::get_list_txt(feature_list, feature_list_path, num_feature);
	dataIO::get_list_txt(learn_name_list, learn_data_name_list_path, num_learn);
	dataIO::get_list_txt(test_name_list, test_data_name_list_path, num_test);

	learn_txt_dir = outdir + "\\learn_feature_list\\";
	test_txt_dir = outdir + "\\test_feature_list\\";
	dataIO::check_folder(learn_txt_dir);
	dataIO::check_folder(test_txt_dir);

	dwknn.create_data_path_list(learn_folder_path, learn_name_list, feature_list, each_learn_txt_path, learn_txt_dir);
	dwknn.create_data_path_list(test_folder_path, test_name_list, feature_list, each_test_txt_path, test_txt_dir);

	/* load learn_mask */
	dataIO::load_matrix(learn_mask, learn_mask_list_path, num_learn);

	/* load learn_tumor */
	dataIO::load_matrix(learn_tumor, learn_tumor_list_path, num_learn);

	/* load test_mask */
	dataIO::load_matrix(test_mask, test_mask_list_path, num_test);

	/* load test_tumor (for optimization) */
	if (argc == 12) {
		dataIO::load_matrix(test_tumor, test_tumor_list_path, num_test);

		element = popcnt::element_count(size_x*size_y);
		for (int i = 0; i < test_tumor.cols(); ++i) {
			unsigned char *A = &test_tumor(0, i);
			unsigned long long *P = new unsigned long long[size_x*size_y];
			popcnt::pack(A, P, size_x*size_y);
			test_data_packed.push_back(P);
		}
	}

	/* load test_data in the mask */
	dwknn.load_features_and_mask_index(each_learn_txt_path, learn_data, learn_mask_index, learn_mask, num_feature);

	/* load test_data in the mask */
	dwknn.load_features_and_mask_index(each_test_txt_path, test_data, test_mask_index, test_mask, num_feature);

	/**
	* @normalization ( learn_data, test_data)
	**/
	for (int i = 0; i < learn_data.size(); ++i) {
		dwknn.normalization(learn_data[i]);
	}

	for (int i = 0; i < test_data.size(); ++i) {
		dwknn.normalization(test_data[i]);
	}

	/* create class_tumor_index and class_musk_index */
	dwknn.create_class_tumor_musk_index(learn_mask_index, learn_tumor, class_tumor_index, class_mask_index);

	/* make learning sample adn learning classified */
	dwknn.make_learning_sample(learning_sample, learn_data, class_tumor_index, class_mask_index, learn_classified, max_num_tumor, max_num_mask, num_feature);

	/* create classified_index */
	for (int i = 0; i < learn_classified.size(); ++i) {
		classified_index.push_back(i);
	}

	std::cout << "Making data base has been finished, starting K-NN >>" << std::endl;

	/* calculate distance and classified */
	ji_vector.clear();
	k_vector.clear();
	result_label.clear();
	distance.clear();
	dataIO::check_folder(outdir + "\\output_image\\");

	/* optimization */
	if (argc == 12) {
		/* process_K_NN */
		if (optimize_k == false) {
			k_vector.push_back(num_k);

			for (int i = 0; i < test_data.size(); ++i) {

				std::cout << "Processing test data: data_" << std::setw(2) << std::setfill('0') << i + 1 << std::endl;
				dwknn.calculate_distance_and_classified(test_data[i], learning_sample, classified_index, distance, test_classified_index, num_feature, num_k);

				dwknn.process_k_nn(distance[i], test_classified_index[i], learn_classified, test_mask_index[i], result_label, num_feature, num_k, size_x, size_y, WKNN);

				unsigned char *B = &result_label[i][0];
				unsigned long long *Q = new unsigned long long[size_x*size_y];
				popcnt::pack(B, Q, size_x*size_y);
				ji.push_back(popcnt::JI(test_data_packed[i], Q, element));

				std::cout << "data_" << std::setw(2) << std::setfill('0') << i + 1 << "_JI = " << ji[i] << std::endl;

				/* memory release */
				delete[]Q;
			}
			ji_vector.push_back(ji);
		}

		if (optimize_k == true) {

			for (int i = 0; i < test_data.size(); ++i) {

				std::cout << "calculate distance : data_" << std::setw(2) << std::setfill('0') << i + 1 << std::endl;
				dwknn.calculate_distance_and_classified(test_data[i], learning_sample, classified_index, distance, test_classified_index, num_feature, num_k);
			}

			for (int index_k = 1; index_k <= num_k; index_k += 2) {
				ji.clear();
				result_label.clear();
				k_vector.push_back(index_k);

				for (int i = 0; i < test_data.size(); ++i) {

					std::cout << "Processing knn : data_" << std::setw(2) << std::setfill('0') << i + 1 << std::endl;
					dwknn.process_k_nn(distance[i], test_classified_index[i], learn_classified, test_mask_index[i], result_label, num_feature, index_k, size_x, size_y, WKNN);

					unsigned char *B = &result_label[i][0];
					unsigned long long *Q = new unsigned long long[size_x*size_y];
					popcnt::pack(B, Q, size_x*size_y);
					ji.push_back(popcnt::JI(test_data_packed[i], Q, element));
					
					std::cout << "data_" << std::setw(2) << std::setfill('0') << i + 1 << "_JI = " << ji[i] << std::endl;
					/* memory release */
					delete[]Q;
				}
				ji_vector.push_back(ji);
			}
		}

		/* save each_K'ji_mean */
		std::ofstream csv_mean_ji(outdir + "\\mean_ji.csv", std::ios::out);
		ji_mean.resize(k_vector.size());
		for (int i = 0; i < k_vector.size(); ++i) {
			ji_mean[i] = std::accumulate(ji_vector[i].begin(), ji_vector[i].end(), 0.0) / (double)ji_vector[i].size();
			std::cout << "ji_mean = " << ji_mean[i] << std::endl;

			csv_mean_ji << k_vector[i] << "," << ji_mean[i] << std::endl;
		}
		ji_max = *std::max_element(ji_mean.begin(), ji_mean.end());
		csv_mean_ji << "max_ji," << ji_max << std::endl;

		csv_mean_ji.close();

		/* save each feature k and ji */
		std::ofstream csv_each_k_ji(outdir + "\\each_k_ji.csv", std::ios::out);
		for (int i = 0; i < k_vector.size(); ++i) {		
			csv_each_k_ji << k_vector[i] << std::endl;
			for (int s = 0; s < num_test; ++s) {
				csv_each_k_ji << ji_vector[i][s] << ",";
			}
			csv_each_k_ji << "\n";
		}

		csv_each_k_ji.close();

	}

	/* for testing */
	if (argc == 11) {
		for (int i = 0; i < test_data.size(); ++i) {

			std::cout << "Processing test data: data_" << std::setw(2) << std::setfill('0') << i + 1 << std::endl;
			dwknn.calculate_distance_and_classified(test_data[i], learning_sample, classified_index, distance, test_classified_index, num_feature, num_k);

			dwknn.process_k_nn(distance[i], test_classified_index[i], learn_classified, test_mask_index[i], result_label, num_feature, num_k, size_x, size_y, WKNN);

		}

	}

	/* save image */
	image_index = 0;
	for (auto s = test_name_list.begin(); s != test_name_list.end(); ++s) {
		imageio.Write(result_label[image_index], outdir + "\\output_image\\" + *s + ".mhd");
		image_index++;
	}

	return EXIT_SUCCESS;
}
