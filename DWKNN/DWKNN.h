#pragma once
#include<string>
#include<vector>
#include<Eigen/Core>
#include<Eigen/Eigenvalues>

class DWKNN {
public:
	DWKNN() {}
	virtual ~DWKNN() {};

	/* create data_path_list */
	template<class T, class T2, class T3, class T4, class T5>
	void create_data_path_list(T input_folder_path, T2 &name_list, T3 &feature_list, T4 &each_data_txt_path, T5 &output_folder_path);

	/* extract label pixel and mask_index (for test_data) */
	template<class T, class T2, class T3, class T4>
	void take_bright_and_mask_index(T input, T2 &bright, T3 &index, T4 mask);

	/* normalization */
	template<class T>
	void normalization(T &input);

	/* load features and mask index (for the pixel in the mask) */
	template<class T, class T2, class T3, class T4>
	void load_features_and_mask_index(T &path_list, T2 &data, T3 &mask_index, T4 &mask, int num_feature);

	/* create class_tumor_index and class_musk_index */
	template<class T, class T2, class T3, class T4>
	void create_class_tumor_musk_index(T &learn_mask_index, T2 &learn_tumor, T3 &class_tumor_index, T4 &class_mask_index);

	/* make learning sample */
	template<class T, class T2, class T3, class T4, class T5>
	void make_learning_sample(T &learning_sample, T2 &learn_data, T3 &class_tumor_index, T4 &class_mask_index, T5 &learning_classified, int max_num_tumor, int max_num_mask, int num_feature);

	/* calculate distance and classified */
	template<class T, class T2, class T3, class T4, class T5>
	void calculate_distance_and_classified(T &single_test_data, T2 &learning_sample, T3 &classified_index, T4 &distance, T5 &test_classified_index, int num_feature, int num_k);

	/* calculate weight */
	template<class T, class T2>
	void calculate_weight(T &distance, T2 &weight, int num_feature);

	/* process K-NN */
	template<class T, class T2, class T3, class T4, class T5>
	void process_k_nn(T &single_distance, T2 &single_test_classified_index, T3 &learn_classified, T4 &test_mask_index, T5 &single_resule_label, int num_feautre, int k, int size_x, int size_y, bool WKNN);

};
