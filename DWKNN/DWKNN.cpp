#include"DWKNN.h"
#include"sort_index.h"
#include<random>
#include<numeric>

typedef double T_learn;
typedef double T_test;
typedef unsigned char T_label;

/* create data_path_list */
template<class T, class T2, class T3, class T4, class T5>
void DWKNN::create_data_path_list(T input_folder_path, T2 &name_list, T3 &feature_list, T4 &each_data_txt_path, T5 &output_folder_path) {
	std::vector<std::string> feature_path;	//path of feature.raw
	std::string feature;					//name of feature.raw (each learning/test data)
	std::string txt_path;					//path of signle learning data

	for (auto i = name_list.begin(); i != name_list.end(); ++i) {
		feature_path.clear();
		for (auto s = feature_list.begin(); s != feature_list.end(); ++s) {
			feature = input_folder_path + "\\" + *s + "\\" + *i + ".raw";
			feature_path.push_back(feature);
		}
		txt_path = output_folder_path + *i + ".txt";
		std::cout << txt_path << std::endl;
		dataIO::write_txt(feature_path, txt_path);
		each_data_txt_path.push_back(txt_path);
	}
}

/* extract brightness and mask_index (for test_data) */
template<class T, class T2, class T3, class T4>
void DWKNN::take_bright_and_mask_index(T input, T2 &bright, T3 &index, T4 mask) {

	std::vector<int> buf;		//buf of index
	buf.clear();
	for (int i = 0; i < mask.size(); ++i) {
		if (mask(i) != 0) {
			buf.push_back(i);
			index.push_back(i);
		}
	}
	bright.resize(buf.size(), input.cols());

	for (int i = 0; i < buf.size(); ++i) {
		bright.row(i) = input.row(buf[i]);
	}
}

/* normalization */
template<class T>
void DWKNN::normalization(T &input) {
	for (int i = 0; i < input.cols(); ++i) {
		double mean = input.col(i).mean();
		double sigma = (input.col(i).array() - mean).square().sum() / input.rows();
		input.col(i) = (input.col(i).array() - mean) / sqrt(sigma);
	}
}

/* load features and mask index (for the pixel in the mask) */
template<class T, class T2, class T3, class T4>
void DWKNN::load_features_and_mask_index(T &path_list, T2 &data, T3 &mask_index, T4 &mask, int num_feature) {
	std::vector<int>buf_mask_index;
	Eigen::Matrix<T_learn, Eigen::Dynamic, Eigen::Dynamic> buf_mat;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> buf_mat_double;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> buf_bright;

	for (int i = 0; i < path_list.size(); ++i) {
		buf_mask_index.clear();
		dataIO::load_matrix(buf_mat, path_list[i], num_feature);
		buf_mat_double = buf_mat.cast<double>();		//change to double
		take_bright_and_mask_index(buf_mat_double, buf_bright, buf_mask_index, mask.col(i));
		data.push_back(buf_bright);
		mask_index.push_back(buf_mask_index);
	}
}

/* create class_tumor_index and class_musk_index */
template<class T, class T2, class T3, class T4>
void DWKNN::create_class_tumor_musk_index(T &learn_mask_index, T2 &learn_tumor, T3 &class_tumor_index, T4 &class_mask_index) {
	std::vector<int> buf_tumor_index;
	std::vector<int> buf_mask_index;

	for (int index_data = 0; index_data < learn_mask_index.size(); ++index_data) {
		buf_tumor_index.clear();
		buf_mask_index.clear();
		for (int j = 0; j < learn_mask_index[index_data].size(); ++j) {

			if (learn_tumor(learn_mask_index[index_data][j], index_data) > 0) {
				buf_tumor_index.push_back(j);
			}
			else buf_mask_index.push_back(j);
		}
		class_tumor_index.push_back(buf_tumor_index);
		class_mask_index.push_back(buf_mask_index);
	}
}


/* make learning sample */
template<class T, class T2, class T3, class T4, class T5>
void DWKNN::make_learning_sample(T &learning_sample, T2 &learn_data, T3 &class_tumor_index, T4 &class_mask_index, T5 &learning_classified, int max_num_tumor, int max_num_mask, int num_feature) {
	std::vector<double> tumor_sample;
	std::vector<double> mask_sample;
	std::vector<std::vector<double>> all_sample;
	std::vector<int> num_tumor;
	std::vector<int> num_mask;
	std::vector<int> index;

	for (int feature_index = 0; feature_index < num_feature; ++feature_index) {
		num_tumor.clear();
		num_mask.clear();
		all_sample.resize(num_feature);

		for (int data_index = 0; data_index < learn_data.size(); ++data_index) {

			tumor_sample.clear();
			/* tumor sample */
			if (class_tumor_index[data_index].size() <= max_num_tumor) {
				for (int k = 0; k < class_tumor_index[data_index].size(); ++k) {
					tumor_sample.push_back(learn_data[data_index](class_tumor_index[data_index][k], feature_index));
				}
			}

			index.clear();
			if (class_tumor_index[data_index].size() > max_num_tumor) {
				index = generateUniqueRandomIntegers(class_tumor_index[data_index].size() - 1, max_num_tumor);

				for (int k = 0; k < index.size(); ++k) {
					tumor_sample.push_back(learn_data[data_index](class_tumor_index[data_index][index[k]], feature_index));
				}
			}

			mask_sample.clear();
			/* mask sample */
			if (class_mask_index[data_index].size() <= max_num_mask) {
				for (int k = 0; k < class_mask_index[data_index].size(); ++k) {
					mask_sample.push_back(learn_data[data_index](class_mask_index[data_index][k], feature_index));
				}
			}

			index.clear();
			if (class_mask_index[data_index].size() > max_num_mask) {
				index = generateUniqueRandomIntegers(class_mask_index[data_index].size() - 1, max_num_mask);
				for (int k = 0; k < index.size(); ++k) {
					mask_sample.push_back(learn_data[data_index](class_mask_index[data_index][index[k]], feature_index));
				}
			}

			num_tumor.push_back((int)tumor_sample.size());		//tumor_number of each learning data
			num_mask.push_back((int)mask_sample.size());		//mask_number of each learning data
			all_sample[feature_index].insert(all_sample[feature_index].end(), tumor_sample.begin(), tumor_sample.end());
			all_sample[feature_index].insert(all_sample[feature_index].end(), mask_sample.begin(), mask_sample.end());
		}

	}
	/* change to eigen matrix */
	learning_sample.resize(all_sample[0].size(), num_feature);
	for (int i = 0; i < num_feature; ++i)
	{
		learning_sample.col(i) = Eigen::Map<Eigen::VectorXd>(&all_sample[i][0], all_sample[i].size());
	}

	/* create learning classified */
	learning_classified.clear();
	for (int i = 0; i < learn_data.size(); ++i) {
		for (int number = 0; number < num_tumor[i]; number++) {
			learning_classified.push_back(1.0);
		}
		for (int number = 0; number < num_mask[i]; number++) {
			learning_classified.push_back(-1.0);
		}
	}
}

/* calculate distance and classified */
template<class T, class T2, class T3, class T4, class T5>
void DWKNN::calculate_distance_and_classified(T &single_test_data, T2 &learning_sample, T3 &classified_index, T4 &distance, T5 &test_classified_index, int num_feature, int num_k) {
	Eigen::MatrixXd buf_distance(learning_sample.rows(), num_feature);
	Eigen::MatrixXd buf_distance_resize(num_k, num_feature);				//for memory release
	Eigen::MatrixXi buf_index_mat(classified_index.size(), num_feature);
	Eigen::MatrixXi buf_index_mat_resize(num_k, num_feature);				//for memory release
	std::vector<int>buf_index;
	std::vector<Eigen::MatrixXd> single_distance;							//each pixel
	std::vector<Eigen::MatrixXi> single_test_classified_index;

	for (int pixel_index = 0; pixel_index < single_test_data.rows(); ++pixel_index) {
		//std::cout << pixel_index << std::endl;

		/* calculate distance */
		for (int i = 0; i < num_feature; ++i) {
			buf_distance.col(i) = learning_sample.col(i).array() - single_test_data(pixel_index, i);
		}
		buf_distance = buf_distance.array().abs();

		/* sorting */
		for (int i = 0; i < num_feature; ++i) {
			buf_index = classified_index;

			std::sort(buf_index.begin(), buf_index.end(), IndexSortCmp(buf_distance.col(i).data(), buf_distance.col(i).data() + buf_distance.col(i).size()));
			std::sort(buf_distance.col(i).data(), buf_distance.col(i).data() + buf_distance.col(i).size());

			buf_index_mat.col(i) = Eigen::Map<Eigen::VectorXi>(&buf_index[0], buf_index.size());
		}

		/* memory release */
		for (int i = 0; i < num_k; ++i) {
			buf_distance_resize.row(i) = buf_distance.row(i);
			buf_index_mat_resize.row(i) = buf_index_mat.row(i);
		}

		single_distance.push_back(buf_distance_resize);

		single_test_classified_index.push_back(buf_index_mat_resize);
	}
	distance.push_back(single_distance);
	test_classified_index.push_back(single_test_classified_index);
}

/* calculate weight */
template<class T, class T2>
void DWKNN::calculate_weight(T &distance, T2 &weight, int num_feature) {

	int last_ele = (int)distance.rows() - 1;
	double k_distance;						//k_th distance
	double first_distance;					//1_st distance
	for (int col = 0; col < num_feature; ++col) {
		for (int row = 0; row < distance.rows(); ++row) {
			if (distance(0, col) != distance(last_ele, col)) {
				k_distance = distance(last_ele, col);
				first_distance = distance(0, col);
				weight(row, col) = ((k_distance - distance(row, col)) / (k_distance - first_distance))*((k_distance + first_distance) / (k_distance + distance(row, col)));
			}
			else(weight(row, col) = 1.0);
		}
	}
}

/* process K-NN */
template<class T, class T2, class T3, class T4, class T5>
void DWKNN::process_k_nn(T &single_distance, T2 &single_test_classified_index, T3 &learn_classified, T4 &test_mask_index, T5 &single_resule_label, int num_feautre, int k, int size_x, int size_y, bool WKNN) {

	Eigen::MatrixXd weight;
	Eigen::MatrixXd buf_classified;
	Eigen::MatrixXd buf;
	std::vector<unsigned char> buf_label(size_x*size_y, 0);

	Eigen::MatrixXd buf_distance(k, num_feautre);
	Eigen::MatrixXi buf_index(k, num_feautre);

	double vote;

	for (int pixel_index = 0; pixel_index < test_mask_index.size(); ++pixel_index) {

		for (int i = 0; i < k; ++i) {
			buf_distance.row(i) = single_distance[pixel_index].row(i);
			buf_index.row(i) = single_test_classified_index[pixel_index].row(i);
		}

		buf_classified.resize(k, num_feautre);
		weight.resize(k, num_feautre);


		if (WKNN == true) {
			calculate_weight(buf_distance, weight, num_feautre);
		}
		if (WKNN == false) {
			weight.fill(1.0);
		}

		for (int i = 0; i < k; ++i) {
			for (int j = 0; j < num_feautre; ++j) {
				buf_classified(i, j) = learn_classified[buf_index(i, j)];
			}
		}

		buf_classified = buf_classified.array()*weight.array();

		buf = buf_classified.rowwise().sum().colwise().sum();

		vote = buf(0, 0);

		if (vote > 0) {
			buf_label[test_mask_index[pixel_index]] = 1;
		}

	}
	single_resule_label.push_back(buf_label);
}

