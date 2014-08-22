//Written by Helen Fan, probably not used in the end.

#ifndef CTSYSTEM_HF_HPP_
#define CTSYSTEM_HF_HPP_

#include "CTSystemElements_HF.h"

class CTSystem
{
public:
	CalibrationParam a_CalParam;
	Detector a_Det;
	ScanParameters a_ScanParam;
	VoxVolParam a_VoxParam;
	ObjectParam a_ObjParam;

	void a_SetCTSystemValue(std::string calibrationfilename, std::string ctparameterfilename);
	void a_SetCTSystemValue(std::string masterconfigfilename);

	~CTSystem(); //delete class
protected:
	CalibrationParam m_CalParam;
	Detector m_Det;
	ScanParameters m_ScanParam;
	VoxVolParam m_VoxParam;
	ObjectParam m_ObjParam;
	
	void m_GetCTSystemValue();
};

void CTSystem::m_GetCTSystemValue()
{
	this->a_CalParam	= this->m_CalParam;
	this->a_Det			= this->m_Det;
	this->a_ScanParam	= this->m_ScanParam;
	this->a_VoxParam	= this->m_VoxParam;
	this->a_ObjParam	= this->m_ObjParam;
}

void CTSystem::a_SetCTSystemValue(std::string calibrationfilename, std::string ctparameterfilename)
{
	this->m_CalParam.LoadFromConfigFile(calibrationfilename);
	this->m_Det.LoadFromConfigFile(ctparameterfilename);
	this->m_ScanParam.LoadFromConfigFile(ctparameterfilename);
	this->m_VoxParam.LoadFromConfigFile(ctparameterfilename);
	this->m_ObjParam.LoadFromConfigFile(ctparameterfilename);
	m_GetCTSystemValue();
	std::cout << "Calibration parameters are loaded from: " << calibrationfilename ;
	std::cout << "CT parameters are loaded from: " << ctparameterfilename << std::endl;
}

void CTSystem::a_SetCTSystemValue(std::string masterconfigfilename)
{
	this->m_CalParam.LoadFromConfigFile(masterconfigfilename);
	this->m_Det.LoadFromConfigFile(masterconfigfilename);
	this->m_ScanParam.LoadFromConfigFile(masterconfigfilename);
	this->m_VoxParam.LoadFromConfigFile(masterconfigfilename);
	this->m_ObjParam.LoadFromConfigFile(masterconfigfilename);
	m_GetCTSystemValue();
	std::cout << "All parameters are loaded from: " << masterconfigfilename <<std::endl;
}

CTSystem::~CTSystem()
{

}

#endif