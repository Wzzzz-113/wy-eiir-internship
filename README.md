##1. 重装系统 
  一般PC用u盘装系统， RK3588 刷好固件 
##2. 插网线链接网络（部署完一定要关代理）
##3. ROS2 安装
参考 fishros 一键安装
wget http://fishros.com/install -O fishros && . fishros
按照如下步骤依次选择安装
[1]:一键安装(推荐):ROS
[1]:更换系统源再继续安装
[2]:更换系统源并清理第三方源  
[1]:humble(ROS2)
[1]:humble(ROS2)桌面版
sudo apt install ros-humble-backward-ros
sudo apt install ros-humble-image-publisher
##4. 安装搜狗输入法(开发者可选)
https://shurufa.sogou.com/linux?r=shouji 
https://shurufa.sogou.com/linux/guide

ps:可能存在搜狗输入法无法添加到键盘的bug
解决方案: 点击输入法键盘config选项->Input Method->点击左下角+号 -> 取消勾选 only show current language选项 ->重新点击左下角+号即可看到搜狗键盘->add后置顶搜狗输入法即可
##5. 安装vscode（开发者可选）
https://code.visualstudio.com/docs/?dv=linux64_deb
##6. 安装git 
sudo apt install git git-lfs
##7. 安装终端终结者 （开发者可选）
参考文档，包含原始终端和终结者终端的界面美化
https://blog.csdn.net/qq_21449473/article/details/139388265
##8. 安装vim
sudo apt-get install vim
##9. 安装openssh-server
sudo apt-get install openssh-server
##10. 安装pip3
  sudo apt-get install python3-pip
##11. 安装远程工具 (开发者可选)
Todesk  向日葵....  官网搜索
##12. 英伟达驱动安装（需要降为535版本）
535指定版本安装 Nvidia drivers
Csdn  https://blog.csdn.net/KRISNAT/article/details/134870009

英伟达驱动安装
ubuntu-drivers devices
sudo apt install nvidia-driver-535

查看设置指令：
  nvidia-smi
  nvidia-settings

Cuda安装：
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin -P .
sudo mv ./cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.2-535.104.05-1_amd64.deb -P .
sudo dpkg -i cuda-repo-ubuntu2204-12-2-local_12.2.2-535.104.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
export PATH=$PATH:/usr/local/cuda-12.2/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.2/lib64
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-12.2
nvcc -V

Cudnn安装（下载好的cudnn 9.3.0）
https://developer.nvidia.com/cudnn
wget https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn-local-repo-ubuntu2204-9.3.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.3.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.3.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update

torch安装
pip3 install torch==2.4.1 torchvision torchaudio --index-url https://pypi.tuna.tsinghua.edu.cn/simple

需要更新onnxruntime-gpu 
pip3 install onnxruntime-gpu --upgrade -i  https://pypi.tuna.tsinghua.edu.cn/simple
