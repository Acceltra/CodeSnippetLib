#pragma once

#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"

namespace ImageProcessSnippetLib
{
	namespace CppImpl
	{

		const int IMAGE_SIZE_512_WIDTH = 512;
		const int IMAGE_SIZE_512_HEIGHT = 512;

		template<class TypeName>
		class DevImage
		{
		public:
			DevImage(int nWidth, int nHeight);
			~DevImage();

			long GetImageSize();

			cudaPitchedPtr pPitchedImage;
			int nImageWidth;
			int nImageHeight;
		};
		typedef DevImage<float> DevFloatImage;

		template<class TypeName>
		class HostImage
		{
		public:
			HostImage(int nWidth, int nHeight);
			~HostImage();

			long GetImageSize();

			TypeName* pHostImage;
			int nBufferLength;
			int nImageWidth;
			int nImageHeight;
		};
		typedef HostImage<float> HostFloatImage;

		class BaseProcessor
		{
		public:
			BaseProcessor();
			~BaseProcessor();

			void CUDAFunctionTimer();
			void CPUFunctionTimer();

			bool ReadFloatImage(DevFloatImage& clsImage, std::string& strFilePath);
			bool ReadFloatImage(HostFloatImage& clsImage, std::string& strFilePath);
			void WriteFloatImage(DevFloatImage& clsImage, std::string& strFilePath);
			void WriteFloatImage(HostFloatImage& clsImage, std::string& strFilePath);

			virtual bool Process();
		};

	}
}
