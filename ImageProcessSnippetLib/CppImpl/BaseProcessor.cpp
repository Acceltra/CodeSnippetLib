#include <fstream>
#include <cstddef>
#include <memory>
#include <type_traits>
#include "BaseProcessor.h"

namespace ImageProcessSnippetLib
{
	namespace CppImpl
	{

		template<class TypeName>
		DevImage<TypeName>::DevImage(int nWidth, int nHeight)
		{
			nImageWidth = nWidth;
			nImageHeight = nHeight;
			pPitchedImage.ptr = nullptr;
			pPitchedImage.pitch = 0;
		}

		template<class TypeName>
		DevImage<TypeName>::~DevImage()
		{

		}

		template<class TypeName>
		HostImage<TypeName>::HostImage(int nWidth, int nHeight)
		{
			nImageWidth = nWidth;
			nImageHeight = nHeight;
			pHostImage = new TypeName[nWidth * nHeight];
			nBufferLength = nWidth * nHeight * sizeof(TypeName);
		}

		template<class TypeName>
		HostImage<TypeName>::~HostImage()
		{
			if (nullptr != pHostImage)
			{
				delete[] pHostImage;
				pHostImage = nullptr;
			}
			nBufferLength = 0;
		}

		BaseProcessor::BaseProcessor()
		{

		}

		BaseProcessor::~BaseProcessor()
		{

		}

		bool BaseProcessor::ReadFloatImage(DevFloatImage& clsImage, std::string& strFilePath)
		{
			std::fstream fIn(strFilePath, std::fstream::in | std::fstream::binary);
			if (!fIn)
			{
				return false;
			}

			int nWidth = clsImage.nImageWidth;
			int nHeight = clsImage.nImageHeight;

			std::unique_ptr<uchar[]> pHostBuffer(new uchar[nWidth * nHeight * sizeof(float)]);
			if (nullptr == pHostBuffer.get())
			{
				fIn.close();
				return false;
			}
			fIn.read((char*)pHostBuffer.get(), nWidth * nHeight * sizeof(float));
			fIn.close();

			cudaMemcpy2D(clsImage.pPitchedImage.ptr, clsImage.pPitchedImage.pitch, pHostBuffer.get(), nWidth * sizeof(float), nWidth * sizeof(float), nHeight, cudaMemcpyHostToDevice);

			return true;
		}

		bool BaseProcessor::ReadFloatImage(HostFloatImage& clsImage, std::string& strFilePath)
		{
			std::fstream fIn(strFilePath, std::fstream::in | std::fstream::binary);
			if (!fIn)
			{
				return false;
			}

			int nWidth = clsImage.nImageWidth;
			int nHeight = clsImage.nImageHeight;

			std::unique_ptr<uchar[]> pHostBuffer(new uchar[nWidth * nHeight * sizeof(float)]);
			if (nullptr == pHostBuffer.get())
			{
				fIn.close();
				return false;
			}
			fIn.read((char*)pHostBuffer.get(), nWidth * nHeight * sizeof(float));
			fIn.close();

			memcpy(clsImage.pHostImage, pHostBuffer.get(), nWidth * nHeight);

			return true;
		}

		void BaseProcessor::WriteFloatImage(DevFloatImage& clsImage, std::string& strFilePath)
		{
			std::fstream fOut(strFilePath, std::fstream::out | std::fstream::binary);
			if (!fOut)
			{
				return;
			}

			int nWidth = clsImage.nImageWidth;
			int nHeight = clsImage.nImageHeight;

			std::unique_ptr<uchar[]> pHostBuffer(new uchar[nWidth * nHeight * sizeof(float)]);
			if (nullptr == pHostBuffer.get())
			{
				fOut.close();
				return;
			}

			cudaMemcpy2D(pHostBuffer.get(), nWidth * sizeof(float), clsImage.pPitchedImage.ptr, clsImage.pPitchedImage.pitch, clsImage.nImageWidth * sizeof(float), clsImage.nImageHeight, cudaMemcpyDeviceToHost);

			fOut.write((char*)pHostBuffer.get(), nWidth * nHeight * sizeof(float));
			fOut.close();
		}

		bool BaseProcessor::Process()
		{
			return true;
		}
	}
}