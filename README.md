# Win32 Caffe
[NOTICE]: Make sure that you have successfully built Win64 based caffe using the project. That will make Win32 version building more easily.
This project is based on [Caffe-rc3](https://github.com/BVLC/caffe/tree/rc3).

## Windows Setup
**Requirements**: Visual Studio 2013. 

Only CPU version is tested under win7 and win10. Precision of probability is lost a bit but the label is right.

I think GPU version would also be built smoothly if you set the environment of CUDA etc. properly. You can try! ^_^

## Steps
0. Rename windows/CommonSettings.props.example to windows/CommonSettings.props.
1. Build Win64 Version. This will automatically and surprisingly download dependencies which Win32 caffe will use.
2. Create Win32 Project Platform copying configurations from x64 for projects "libcaffe" and "classification" respectively.
3. Switch Project Platform to Win32 and rebuilt for projects "libcaffe" and "classification".
4. Check it out using classification.exe.

## Contributor
team: [VisionRush](https://github.com/VisionRush)  
authors: WangJian(wangjian@ia.ac.cn), [WangBo(wangbo@ia.ac.cn)](https://github.com/wangboNlpr)


# Windows Caffe

**This is an experimental, Microsoft-led branch by Pavle Josipovic (@pavlejosipovic). It is a work-in-progress.**

This branch of Caffe ports the framework to Windows.

[![Travis Build Status](https://api.travis-ci.org/BVLC/caffe.svg?branch=windows)](https://travis-ci.org/BVLC/caffe) Travis (Linux build)

[![Build status](https://ci.appveyor.com/api/projects/status/128eg95svel2a2xs?svg=true)]
(https://ci.appveyor.com/project/pavlejosipovic/caffe-v45qi) AppVeyor (Windows build)

## Windows Setup
**Requirements**: Visual Studio 2013

### Pre-Build Steps
Copy `.\windows\CommonSettings.props.example` to `.\windows\CommonSettings.props`

By defaults Windows build requires `CUDA` and `cuDNN` libraries.
Both can be disabled by adjusting build variables in `.\windows\CommonSettings.props`.
Python support is disabled by default, but can be enabled via `.\windows\CommonSettings.props` as well.
3rd party dependencies required by Caffe are automatically resolved via NuGet.

### CUDA
Download `CUDA Toolkit 7.5` [from nVidia website](https://developer.nvidia.com/cuda-toolkit).
If you don't have CUDA installed, you can experiment with CPU_ONLY build.
In `.\windows\CommonSettings.props` set `CpuOnlyBuild` to `true` and set `UseCuDNN` to `false`.

### cuDNN
Download `cuDNN v3` or `cuDNN v4` [from nVidia website](https://developer.nvidia.com/cudnn).
Unpack downloaded zip to %CUDA_PATH% (environment variable set by CUDA installer).
Alternatively, you can unpack zip to any location and set `CuDnnPath` to point to this location in `.\windows\CommonSettings.props`.
`CuDnnPath` defined in `.\windows\CommonSettings.props`.
Also, you can disable cuDNN by setting `UseCuDNN` to `false` in the property file.

### Python
To build Caffe Python wrapper set `PythonSupport` to `true` in `.\windows\CommonSettings.props`.
Download Miniconda 2.7 64-bit Windows installer [from Miniconda website] (http://conda.pydata.org/miniconda.html).
Install for all users and add Python to PATH (through installer).

Run the following commands from elevated command prompt:

```
conda install --yes numpy scipy matplotlib scikit-image pip
pip install protobuf
```

#### Remark
After you have built solution with Python support, in order to use it you have to either:  
* set `PythonPath` environment variable to point to `<caffe_root>\Build\x64\Release\pycaffe`, or
* copy folder `<caffe_root>\Build\x64\Release\pycaffe\caffe` under `<python_root>\lib\site-packages`.

### Matlab
To build Caffe Matlab wrapper set `MatlabSupport` to `true` and `MatlabDir` to the root of your Matlab installation in `.\windows\CommonSettings.props`.

#### Remark
After you have built solution with Matlab support, in order to use it you have to:
* add the generated `matcaffe` folder to Matlab search path, and
* add `<caffe_root>\Build\x64\Release` to your system path.

### Build
Now, you should be able to build `.\windows\Caffe.sln`

## Further Details

Refer to the BVLC/caffe master branch README for all other details such as license, citation, and so on.
