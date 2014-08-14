//Got it from Jared Moore, edited by Helen Fan

#ifndef RAWDATA_HPP_
#define RAWDATA_HPP_


template <class T>
class SaveRawData
{
public:
	SaveRawData(){};
	
	void print();
	void operator ()(T* DataPtr, unsigned long long MemSize, std::string FileName);
	void operator ()(T** DataPrt, unsigned long long MemSize1, unsigned long long Memsize2, std::string FileName);
	~SaveRawData(){};

private:
	std::string p_filename;
};

template <class T>
void SaveRawData<T>::operator ()(T *DataPtr, unsigned long long MemSize, std::string FileName)
{
	std::ofstream fpRawData;
	fpRawData.open(FileName.c_str(), std::ios::binary);
	fpRawData.write((char*)DataPtr,MemSize);
	fpRawData.close();
	this->p_filename = FileName;
}

template <class T>
void SaveRawData<T>:: operator ()(T **DataPtr, unsigned long long MemSize1, unsigned long long MemSize2, std::string FileName)
{
	std::ofstream fpRawData;
	fpRawData.open(FileName.c_str(), std::ios::binary);
	
	for (int i = 0; i < (int)MemSize1; i++)
	{
		fpRawData.write((char*)DataPtr[i],MemSize2);	
	}
	fpRawData.close();
	this->p_filename = FileName;
}

template <class T>
void SaveRawData<T>::print()
{
	std::cout<<"Raw data written to "<<this->p_filename<<"\n";
}

template <class T>
class LoadRawData
{
public:
	LoadRawData(){};
	void operator()(T* DataPtr, unsigned long long MemSize, std::string FileName);
	void operator()(T** DataPtr, unsigned long long MemSize1, unsigned long long MemSize2, std::string FileName);
	void print();
	~LoadRawData(){};

private:
	std::string p_filename;

};

template <class T>
void LoadRawData<T>::operator ()(T *DataPtr, unsigned long long MemSize, std::string FileName)
{
	std::ifstream fpRawData;
	fpRawData.open(FileName.c_str(), std::ios::in | std::ios::binary);

	if(! fpRawData.is_open())
	{std::cout<<"Error loading raw data" <<std::endl;}

	fpRawData.read((char*)DataPtr,MemSize);
	if(fpRawData.gcount()==0)
	{std::cout<< "Error: 0 bytes read for "<<FileName <<std::endl;}

	this->p_filename = FileName;
	fpRawData.close();
}

template <class T>
void LoadRawData<T>::operator ()(T **DataPtr, unsigned long long MemSize1, unsigned long long MemSize2, std::string FileName)
{
	std::ifstream fpRawData;
	fpRawData.open(FileName.c_str(), std::ios::in | std::ios::binary);
	if(! fpRawData.is_open())
	{std::cout<<"Error load raw data" << std::endl;}

	for (unsigned int i = 0; i < MemSize1; i++)
	{
		fpRawData.read((char*)DataPtr[i], MemSize2);
		if(fpRawData.gcount()==0)
		{std::cout<< "Error: 0 bytes read for "<<FileName <<std::endl;}

	}
	this->p_filename = FileName;
	fpRawData.close();
}

template <class T>
void LoadRawData<T>::print()
{
	std::cout <<"Raw data loaded from " << this->p_filename << std::endl;
}

#endif