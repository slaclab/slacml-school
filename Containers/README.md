## Install Vagrant

### On Ubuntu systems

https://www.vagrantup.com/downloads.html

```
sudo apt install virtualbox
dpkg install vagrant_2.2.7_x86_64.deb
```

### On RHEL/CentOS systems

```
sudo curl http://download.virtualbox.org/virtualbox/rpm/rhel/virtualbox.repo > /etc/yum.repo.d/virtualbox.repo
yum update && yum install VirtualBox-5.1
yum install vagrant_2.2.7_linux_amd64.zip
```


### On Mac

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew cask install virtualbox
brew cask install vagrant
```


## Start a CentOS7 Virtual Machine

```
mkdir centos7
cd centos7
vagrant init centos/7
vagrant up
```

// vagrant init sylabs/singularity-3.5-centos-7-64


## SSH into the VM

```
vagrant ssh
ls /vagrant # directory from host OS shared here
```

## Install docker and singularity

```
sudo -s
yum install -y epel-release
yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
yum update -y
yum install -y singularity
yum install -y docker-ce && systemctl start docker
```

## Exit SSH from VM

Logout:

```
^d^d
```

## Snapshot VM

```
vagrant snapshot save clean-install
```

## Destroy VM (from host)

```
vagrant destroy
```



## Pull an existing Docker image and Run in singularity

# Pull the image from dockerhub

```
singularity pull docker://pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime
```

# Check for the image on pwd

```
ls pytorch_1.5-cuda10.1-cudnn7-runtime.sif
```


# Run the container

```
singularity exec pytorch_1.5-cuda10.1-cudnn7-runtime.sif python [args…]
```

# Enter the container (bring up shell within container)

```
singularity shell pytorch_1.5-cuda10.1-cudnn7-runtime.sif
```


## Add Uproot python Library into Container

### Edit Dockerfile

```
$ cat Dockerfile
FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime
RUN pip install uproot
```

### Build the image using Docker

```
sudo docker build . -t yee379/pytorch-test:1.0
```

### Push the image to Dockerhub

Login to docker using Dockerhub account

```
sudo docker login
```

Upload new container image to Dockerhub

```
sudo docker push yee379/pytorch-test:1.0
```

### Run new Image on singularity

```
singularity pull docker://slaclab/pytorch-test:1.0
singularity exec  pytorch-test_latest.sif python
...
>>> import uproot
```


## Run on SDF

### Get an interactive session via slurm

```
ssh ytl@sdf-login.slac.stanford.edu
srun --partition=ml --pty bash
```

### setup singularity environment to use different cache dir

```
export DIR=$LSCRATCH
export SINGULARITY_LOCALCACHEDIR=$DIR
export SINGULARITY_CACHEDIR=$DIR
export SINGULARITY_TMPDIR=$DIR
```

### Pull the image

```
singularity pull docker://slaclab/pytorch-test:1.0
```

### Run the container on SDF

```
singularity exec  pytorch-test_latest.sif python
```


