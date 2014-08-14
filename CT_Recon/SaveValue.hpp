#ifndef SAVEVALUE_HPP_
#define SAVEVALUE_HPP_

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

struct SaveValue
{
	void operator()(std::string FileName, float Value)
	{
		std::ofstream fpFile(FileName.c_str(), std::ios::app);
		if(! fpFile)
		{std::cout<< "Error opening output file" <<std::endl;}
		
		std::stringstream sstr;
		sstr<<Value<<"\t";
		fpFile<<sstr.str();
		fpFile.close();
	}
};

struct Save2Values
{
	void operator()(std::string FileName, float Value1, double Value2)
	{
		std::ofstream fpFile(FileName.c_str(), std::ios::app);
		if(! fpFile)
		{std::cout<<"Error opening output file: Save2Values" <<std::endl;}

		std::stringstream sstr;
		sstr<<Value1<<"\t"<<Value2<<std::endl;
		fpFile<<sstr.str();
		fpFile.close();
	}
};

#endif