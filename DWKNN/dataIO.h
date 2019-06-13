#pragma once

/**
* @dataIO
*
* @Author Zhihui Lu
* @Sponsor Atsushi Saito
* @date 2018/06/01
**/

/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
* @Attention: The type of defined Matrix/Vector must be the same as the type of input/output data.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

/**
* @Functions:
*
* @Load Matrix
* @Load Vector
* @Write Binary data
* @Check Folder
**/

#include<fstream>
#include<string>
#include<iostream>
#include<list>
#include<direct.h>
#include <windows.h>
#include <imagehlp.h>
#pragma comment(lib, "imagehlp.lib")
#include <Shlwapi.h>
#pragma comment(lib, "ShLwApi.lib")


namespace dataIO {

	/* replace string */
	void replace_str(std::string &str, const std::string &search_str, const std::string &replace_str) {

		while (1)
		{
			std::string::size_type pos;
			pos = str.find(search_str);
			if (pos == std::string::npos)break;
			str.replace(pos, search_str.length(), replace_str, 0, std::string::npos);
		}
	}

	/* erase extension of filename */
	std::string erase_exten(const std::string &path)
	{
		std::string::size_type pos = path.find_last_of("\\/");
		if (pos == std::string::npos) return "error directory";
		else return path.substr(0, pos);
	}

	/* count number of text lines */
	template<class T>
	long count_number_of_text_lines(T &path) {
		long i = 0;

		std::ifstream ifs(path, std::ios::in);
		//fail to open file
		if (!ifs) {
			std::cerr << "Cannot open file: " << path << std::endl;
			std::abort();
		}
		else {
			std::string line;
			while (true) {
				getline(ifs, line);

				//skip empty strings
				if (line.length()) {
					i++;
				}
				if (ifs.eof()) {
					break;
				}
			}
		}
		return i;
	}

	/**
	* @check folder
	* @can also create a folder tree
	* @Usage: [path]
	**/
	void check_folder(std::string path) {
		replace_str(path, "/", "\\");
		if (PathIsDirectory(path.c_str()) == 0) {
			MakeSureDirectoryPathExists(path.c_str());
			std::cout << "Directory has been created!" << std::endl;
		}
	}

	/* get list txt */
	void get_list_txt(std::list<std::string> &list_txt, std::string path_list, int num) {
		list_txt.clear();
		std::ifstream ifs(path_list, std::ios::in);
		//fail to open file
		if (!ifs) {
			std::cerr << "Cannot open file: " << path_list << std::endl;
			std::abort();
		}
		else {
			std::string buf;
			while (std::getline(ifs, buf)) {
				//skip empty strings
				if (buf.length()) {
					list_txt.push_back(buf);
				}
			}
		}
		list_txt.resize(num);
		ifs.close();
	}

	/* get file size */
	long get_file_size(std::string filename) {
		FILE *fp;
		struct stat st;
		if (fopen_s(&fp, filename.c_str(), "rb") != 0) {
			std::cerr << "Cannot open file: " << filename << std::endl;
			std::abort();
		}
		fstat(_fileno(fp), &st);
		fclose(fp);
		return st.st_size;
	}

	/**
	* @Load Matrix
	* @The type of defined Matrix/Vector must be the same as the type of input/output data.
	* 
	* @Usage: [Matrix] [list_path] [number of data]  
	**/
	template<class T>
	void load_matrix(T &matrix, const std::string data_list_txt, int num) {

		/* get file list */
		std::list<std::string> file_list;
		get_list_txt(file_list, data_list_txt, num);

		/* load bin */
		const std::size_t num_of_data = file_list.size();
		int dimX = get_file_size(*file_list.begin()) / sizeof(decltype(matrix(0, 0)));
		std::cout << "Matrix dimension = " << dimX << std::endl;

		/* data_load */
		matrix.resize(dimX, num_of_data);
		{
			FILE *fp;
			decltype(&matrix(0, 0)) x = &matrix(0, 0);
			for (auto s = file_list.begin(); s != file_list.end(); ++s) {
				fopen_s(&fp, (*s).c_str(), "rb");
				x += fread(x, sizeof(decltype(matrix(0, 0))), dimX, fp);
				fclose(fp);
			}
		}
	}
	
	/**
	* @Load Vector
	* @The type of defined Matrix/Vector must be the same as the type of input/output data.
	* @Compatible with Eigen::vector, std::vector
	*
	* @Usage: [Vector] [list_path] [number of elements]
	**/
	template<class T>
	void load_vector(T &vector, const std::string data_list_txt, int num) {

		/* get file list */
		std::list<std::string> file_list;
		get_list_txt(file_list, data_list_txt, num);
		std::string file_name = file_list.front();

		FILE *fp;
		if (fopen_s(&fp, file_name.c_str(), "rb") != 0) {
			std::cerr << "Cannot open file: " << file_name << std::endl;
			std::abort();
		}

		vector.resize(num);
		fread(vector.data(), sizeof(decltype(vector[0])), num, fp);
		std::cout << "Vector dimension = " << vector.size() << std::endl;
	}

	/**
	* @Write Binary data {(.raw); (.vect)}
	*
	* @The type of defined Matrix/Vector must be the same as the type of input/output data.
	* @usage: [Vector] [file_name] [num of elements]
	**/
	template< class T >
	void write_bin(T vector, std::string file_name, size_t num) {

		/* check pass */
		std::string path = erase_exten(file_name) + "\\";
		check_folder(path.c_str());
		
		/* write */
		FILE *fp;
		decltype(&vector[0]) x = &vector[0];
		fopen_s(&fp, file_name.c_str(), "wb");
		fwrite(x, sizeof(decltype(vector[0])), num, fp);
		fclose(fp);
	}

	/**
	* @Write vector to txt
	**/
	template<class T>
	void write_txt(T &vector, std::string path) {
		std::ofstream out(path, std::ios::out);
		for (int i = 0; i < vector.size(); ++i) {
			out << vector[i] << std::endl;
		}
		out.close();
	}
};
