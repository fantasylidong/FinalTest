#include <io.h>
#include <iostream>
#include <algorithm>
#include <string>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

//读取指定目录下所有图片文件，目前只读取当前目录不读取下一级目录
vector<string> readfile(string srcpath) {
	vector<string> flist; //文件+文件名
	try {
		for (int i = 0; i < srcpath.size(); i++) {
			if (srcpath[i] == '\\')
			{
				srcpath.insert(i, "\\");
				i++;
			}
	    }
		cout << srcpath;
		const char *filepath = srcpath.c_str();
		intptr_t hFile;
		size_t n;//无符号整型
		string p, t;
		struct _finddata_t fileinfo;
		//_findfirst失败返回-1
		if ((hFile = _findfirst(p.assign(filepath).append("\\*").c_str(), &fileinfo)) != -1) {
			do {
				if (!(fileinfo.attrib & _A_SUBDIR)) {
					p.assign(filepath).append("\\").append(fileinfo.name);
					flist.push_back(p);//先放入文件
					flist.push_back(fileinfo.name);//在放入文件名
				}
			} while (_findnext(hFile, &fileinfo) == 0);
			_findclose(hFile);
		}
	}
	catch (std::exception &e) {
		cout << e.what() << endl;
	}
	return flist;
}

//修改图片大小，并保存在指定位置
void myresize(vector<string> flist,Size dst_size,string dst_dir) {
	int n = flist.size();
	for (int i = 0;i < n;i = i+2)
	{
		Mat iimg = imread(flist[i]);
		Mat oimg;
		if (iimg.empty())
		{
			cout << "读取文件" << flist[i] << "失败" << endl;
			break;
		}
		resize(iimg, oimg, dst_size); //重定义目标图片大小

		imwrite(dst_dir+"\\"+ flist[i+1], oimg); //保存图片
	}
}

int main(int arg,char ** argv)
{
	//输入说明 srcdir width height dst_dir,无法对gif处理
	//srcdir:形如C:\Users\hujing\Desktop\srcimage
	//width,height 整数
	//dst_dir:形如 C:\Users\hujing\Desktop\srcimage

	int width, height;
	string src_dir, dst_dir;
	cin >> src_dir >> width >> height >> dst_dir;
	Size dst_size(width,height);
	vector<string> flist = readfile(src_dir);
	myresize(flist,dst_size,dst_dir);
	return 0;
}