---
title: Ubuntu+linuxbrew+zsh+oh-my-zsh
date: 2020-01-11 20:52:42
categories:
- Tips
tags:
- Ubuntu
- Linuxbrew
- Zsh
- Oh-my-zsh
---

参考：

> [Homebrew Documentation](https://docs.brew.sh/Homebrew-on-Linux)
> [Oh My ZSH!](https://ohmyz.sh/)
> [Installing ZSH](https://github.com/ohmyzsh/ohmyzsh/wiki/Installing-ZSH)

租一个服务器能干什么，我想大致分为两个方面：使用和学习。可以将云盘部署到服务器上，那么你就拥有了一个私有云盘，也可以将WordPress部署到服务器上，那么你就拥有了一个可以写博客的个人网站；学习JavaWeb，需要了解Tomcat、MySQL、Nginx、Redis等等，你可以在服务器上运行这些程序而不必使用宝贵的本地资源。为了后续的使用以及学习，首先需要优化一下我们的服务器配置，因为控制服务器一般都是通过命令行，所以前期优化一下，后续会更好操作。

<!-- more -->

## 1. 为什么需要标题中的工具

首先你需要一个云服务器，可以是华为云、腾讯云、阿里云等等，注册购买即可，最便宜的1核心CPU加2GB内存加40GB/50GB的硬盘存储基本足够前期的学习使用，你可以先租1个月玩玩，如果有学生优惠基本上一个月只要10元。云服务器的操作系统一般有Windows Server、CentOS以及Ubuntu等等，我最常用的是Ubuntu而且出了问题能够最容易找到解决方法，稳定性什么的以后再说。Ubuntu版本可以直接上18.04，不用考虑旧版本16.04。

得到服务器后我会先安装zsh，zsh就是命令行工具，我们在服务器上输入的指令都通过zsh执行，zsh搭配oh-my-zsh可以实现非常舒适的UI效果并且提供一些很有用的插件，比如自动提示命令等等。

然后安装Linuxbrew，这是一个包管理工具，是从Homebrew迁移而来（Homebrew只能在macOS上使用），在服务器上安装了brew之后，后续需要安装的软件就都可以通过brew安装，如果brew仓库没有再考虑自行安装。使用brew最明显的好处是你可以直接通过brew更新、删除、查看所安装的软件，而且brew可以提供快速开启某些服务的命令，比如MySQL、Redis、Tomcat等等。

## 2. Ubuntu服务器配置

我的服务器是Ubuntu18.04，拿到服务器后首先可以ssh登录上服务器，购买之后会给你用户名和首次登录密码，有些厂商给的是root用户有些是其他用户例如ubuntu

```shell
ssh ubuntu@175.24.47.141
```
第一次登录需要输入密码

{% asset_img ssh.png %}

登录成功结果

{% asset_img login.png %}

一般来说初始的服务器是不支持中文的，此时你将输入法调为中文也是无法打字的，而且某些文件如果里面包含中文，则会显示为乱码或者问号，所以第一步我会先配置中文支持。

### 2.1 中文支持

先给出我写好的脚本，在服务器任意目录保存为`ch.sh`，然后执行`chmod 777 ch.sh`，最后执行`sh ch.sh`，服务器会重启，之后再登录就可以用中文打字了，而且中文不会乱码。

```sh
echo "------Start to support Chinese------"
echo "------Install Chinese language pack------"
sudo apt-get install language-pack-zh-hans -y

echo "------Set environment------"
sudo sed -i '$aLANG=zh_CN.UTF-8\nLANGUAGE=zh_CN:zh:en_US:en\nLC_CTYPE="en_US.UTF-8"\nLC_ALL=en_US.UTF-8' /etc/environment

sudo locale-gen

echo "------Reboot------"
sudo reboot
```

然后再说明一下脚本里都干了些什么，首先`sudo apt-get install language-pack-zh-hans -y`安装中文支持包，然后执行

```shell
sudo sed -i '$aLANG=zh_CN.UTF-8\nLANGUAGE=zh_CN:zh:en_US:en\nLC_CTYPE="en_US.UTF-8"\nLC_ALL=en_US.UTF-8 /etc/environment
```

在`/etc/environment`末尾添加以下几行配置，这个是永久设置环境变量

```sh
LANG=zh_CN.UTF-8
LANGUAGE=zh_CN:zh:en_US:en
LC_CTYPE="en_US.UTF-8"
LC_ALL=en_US.UTF-8
```

最后是`sudo locale-gen`编译生成编码相关文件，然后`sudo reboot`重启。

### 2.2 ssh免密登录

每次ssh登录都需要输入密码，这是非常烦人的事情，可以配置ssh免密登录，需要本机和服务器做一个联动，你在本机生成一个key，然后将key保存到服务器上的某个位置，之后再从本机ssh服务器时，服务器就知道了是从哪个机器访问的服务器，如果有对应的key就直接让你连接，否则需要密码。

本机输入指令`ssh-keygen -t rsa`，然后你就可以在`/home/usera/.ssh/id_rsa.pub`中查看生成的密钥，注意这个目录是在指令执行了输出结果里出现的，不是所有人都一样

```shell
[usera@local ~]$ ssh-keygen -t rsa
Generating public/private rsa key pair.
Enter file in which to save the key (/home/usera/.ssh/id_rsa): 
Created directory '/home/usera/.ssh'.
Enter passphrase (empty for no passphrase): 
Enter same passphrase again: 
Your identification has been saved in /home/usera/.ssh/id_rsa.
Your public key has been saved in /home/usera/.ssh/id_rsa.pub.
The key fingerprint is:
39:f2:fc:70:ef:e9:bd:05:40:6e:64:b0:99:56:6e:01 usera@local
The key's randomart image is:
+--[ RSA 2048]----+
|          Eo*    |
|           @ .   |
|          = *    |
|         o o .   |
|      . S     .  |
|       + .     . |
|        + .     .|
|         + . o . |
|          .o= o. |
+-----------------+
```

我们需要的是`id_rsa.pub`的内容，以`ssh-rsa`开头的文本，将`id_rsa.pub`的内容复制一份放到服务器的根目录的`.ssh/authorized_keys`文件中，如果没有`.ssh`目录及`authorized_keys`文件，那就创建一份，之后ssh就不需要密码了。

```pub
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDG79mJLYz80Q+kh7MNUH4uLc/sBRyJPQqjOoSEA/co2XXXXXXXXjrxcLoKySsGchi/zALeo9aTaNZSn8nNwaIcg/S+yxZeB6XuqJhjWxQGOonRbAPPcnOldxk/S0J4WS+cFbp0gCmBuu17fjaQXXXXXXXXXXXXXXXXXXXXXXXXXX+N9YXXXXXXXXXXXXXXXXm37ArgxfKoh5U0W2pZhDDdeHeriK5oPu/D8ZN36RVMQ/kxUnuA+Kpv35MjboAjPsT6sa+RnsT/Ftg/ZQXOMV/Tz7UQa7vOERjFoTzMidHhwztZuOw/cTNpDozextbPGBxoWb7rpA0sMNLNoPAX XXXXXXXXXXX
```

## 3. zsh&oh-my-zsh安装

### 3.1 zsh

接下来安装zsh和oh-my-zsh

```shell
sudo apt-get install zsh
```

然后设置默认为zsh

```shell
chsh -s /bin/zsh
```

重新ssh登录，会提示需要完成zsh配置，这里选`1`即可

```shell
This is the Z Shell configuration function for new users,
zsh-newuser-install.
You are seeing this message because you have no zsh startup files
(the files .zshenv, .zprofile, .zshrc, .zlogin in the directory
~).  This function can help you with a few settings that should
make your use of the shell easier.

You can:

(q)  Quit and do nothing.  The function will be run again next time.

(0)  Exit, creating the file ~/.zshrc containing just a comment.
     That will prevent this function being run again.

(1)  Continue to the main menu.

(2)  Populate your ~/.zshrc with the configuration recommended
     by the system administrator and exit (you will need to edit
     the file by hand, if so desired).

--- Type one of the keys in parentheses ---
```

然后会出现一些初始配置，这里直接选`0`即可

```shell
Please pick one of the following options:

(1)  Configure settings for history, i.e. command lines remembered
     and saved by the shell.  (Recommended.)

(2)  Configure the new completion system.  (Recommended.)

(3)  Configure how keys behave when editing command lines.  (Recommended.)

(4)  Pick some of the more common shell options.  These are simple "on"
     or "off" switches controlling the shell's features.

(0)  Exit, creating a blank ~/.zshrc file.

(a)  Abort all settings and start from scratch.  Note this will overwrite
     any settings from zsh-newuser-install already in the startup file.
     It will not alter any of your other settings, however.

(q)  Quit and do nothing else.  The function will be run again next time.
--- Type one of the keys in parentheses ---
```

最后安装完成的结果

```shell
The function will not be run in future, but you can run
it yourself as follows:
  autoload -Uz zsh-newuser-install
  zsh-newuser-install -f

The code added to ~/.zshrc is marked by the lines
# Lines configured by zsh-newuser-install
# End of lines configured by zsh-newuser-install
You should not edit anything between these lines if you intend to
run zsh-newuser-install again.  You may, however, edit any other part
of the file.
# 此时你的用户名会改变，且UI有变化
VM-0-9-ubuntu% 
```

### 3.2 oh-my-zsh

然后安装oh-my-zsh，这里可能出现443问题`curl: (7) Failed to connect to raw.github.com port 443: Connection refused`

```shell
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

翻墙查看`https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh`的内容，并将内容保存为`install.sh`，然后执行`sh install.sh`即可完成安装。

```sh
#!/bin/sh
#
# This script should be run via curl:
#   sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
# or wget:
#   sh -c "$(wget -qO- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
#
# As an alternative, you can first download the install script and run it afterwards:
#   wget https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh
#   sh install.sh
#
# You can tweak the install behavior by setting variables when running the script. For
# example, to change the path to the Oh My Zsh repository:
#   ZSH=~/.zsh sh install.sh
#
# Respects the following environment variables:
#   ZSH     - path to the Oh My Zsh repository folder (default: $HOME/.oh-my-zsh)
#   REPO    - name of the GitHub repo to install from (default: ohmyzsh/ohmyzsh)
#   REMOTE  - full remote URL of the git repo to install (default: GitHub via HTTPS)
#   BRANCH  - branch to check out immediately after install (default: master)
#
# Other options:
#   CHSH    - 'no' means the installer will not change the default shell (default: yes)
#   RUNZSH  - 'no' means the installer will not run zsh after the install (default: yes)
#
# You can also pass some arguments to the install script to set some these options:
#   --skip-chsh: has the same behavior as setting CHSH to 'no'
#   --unattended: sets both CHSH and RUNZSH to 'no'
# For example:
#   sh install.sh --unattended
#
set -e

# Default settings
ZSH=${ZSH:-~/.oh-my-zsh}
REPO=${REPO:-ohmyzsh/ohmyzsh}
REMOTE=${REMOTE:-https://github.com/${REPO}.git}
BRANCH=${BRANCH:-master}

# Other options
CHSH=${CHSH:-yes}
RUNZSH=${RUNZSH:-yes}


command_exists() {
 command -v "$@" >/dev/null 2>&1
}

error() {
 echo ${RED}"Error: $@"${RESET} >&2
}

setup_color() {
 # Only use colors if connected to a terminal
 if [ -t 1 ]; then
  RED=$(printf '\033[31m')
  GREEN=$(printf '\033[32m')
  YELLOW=$(printf '\033[33m')
  BLUE=$(printf '\033[34m')
  BOLD=$(printf '\033[1m')
  RESET=$(printf '\033[m')
 else
  RED=""
  GREEN=""
  YELLOW=""
  BLUE=""
  BOLD=""
  RESET=""
 fi
}

setup_ohmyzsh() {
 # Prevent the cloned repository from having insecure permissions. Failing to do
 # so causes compinit() calls to fail with "command not found: compdef" errors
 # for users with insecure umasks (e.g., "002", allowing group writability). Note
 # that this will be ignored under Cygwin by default, as Windows ACLs take
 # precedence over umasks except for filesystems mounted with option "noacl".
 umask g-w,o-w

 echo "${BLUE}Cloning Oh My Zsh...${RESET}"

 command_exists git || {
  error "git is not installed"
  exit 1
 }

 if [ "$OSTYPE" = cygwin ] && git --version | grep -q msysgit; then
  error "Windows/MSYS Git is not supported on Cygwin"
  error "Make sure the Cygwin git package is installed and is first on the \$PATH"
  exit 1
 fi

 git clone -c core.eol=lf -c core.autocrlf=false \
  -c fsck.zeroPaddedFilemode=ignore \
  -c fetch.fsck.zeroPaddedFilemode=ignore \
  -c receive.fsck.zeroPaddedFilemode=ignore \
  --depth=1 --branch "$BRANCH" "$REMOTE" "$ZSH" || {
  error "git clone of oh-my-zsh repo failed"
  exit 1
 }

 echo
}

setup_zshrc() {
 # Keep most recent old .zshrc at .zshrc.pre-oh-my-zsh, and older ones
 # with datestamp of installation that moved them aside, so we never actually
 # destroy a user's original zshrc
 echo "${BLUE}Looking for an existing zsh config...${RESET}"

 # Must use this exact name so uninstall.sh can find it
 OLD_ZSHRC=~/.zshrc.pre-oh-my-zsh
 if [ -f ~/.zshrc ] || [ -h ~/.zshrc ]; then
  if [ -e "$OLD_ZSHRC" ]; then
   OLD_OLD_ZSHRC="${OLD_ZSHRC}-$(date +%Y-%m-%d_%H-%M-%S)"
   if [ -e "$OLD_OLD_ZSHRC" ]; then
    error "$OLD_OLD_ZSHRC exists. Can't back up ${OLD_ZSHRC}"
    error "re-run the installer again in a couple of seconds"
    exit 1
   fi
   mv "$OLD_ZSHRC" "${OLD_OLD_ZSHRC}"

   echo "${YELLOW}Found old ~/.zshrc.pre-oh-my-zsh." \
    "${GREEN}Backing up to ${OLD_OLD_ZSHRC}${RESET}"
  fi
  echo "${YELLOW}Found ~/.zshrc.${RESET} ${GREEN}Backing up to ${OLD_ZSHRC}${RESET}"
  mv ~/.zshrc "$OLD_ZSHRC"
 fi

 echo "${GREEN}Using the Oh My Zsh template file and adding it to ~/.zshrc.${RESET}"

 cp "$ZSH/templates/zshrc.zsh-template" ~/.zshrc
 sed "/^export ZSH=/ c\\
export ZSH=\"$ZSH\"
" ~/.zshrc > ~/.zshrc-omztemp
 mv -f ~/.zshrc-omztemp ~/.zshrc

 echo
}

setup_shell() {
 # Skip setup if the user wants or stdin is closed (not running interactively).
 if [ $CHSH = no ]; then
  return
 fi

 # If this user's login shell is already "zsh", do not attempt to switch.
 if [ "$(basename "$SHELL")" = "zsh" ]; then
  return
 fi

 # If this platform doesn't provide a "chsh" command, bail out.
 if ! command_exists chsh; then
  cat <<-EOF
   I can't change your shell automatically because this system does not have chsh.
   ${BLUE}Please manually change your default shell to zsh${RESET}
EOF
  return
 fi

 echo "${BLUE}Time to change your default shell to zsh:${RESET}"

 # Prompt for user choice on changing the default login shell
 printf "${YELLOW}Do you want to change your default shell to zsh? [Y/n]${RESET} "
 read opt
 case $opt in
  y*|Y*|"") echo "Changing the shell..." ;;
  n*|N*) echo "Shell change skipped."; return ;;
  *) echo "Invalid choice. Shell change skipped."; return ;;
 esac

 # Check if we're running on Termux
 case "$PREFIX" in
  *com.termux*) termux=true; zsh=zsh ;;
  *) termux=false ;;
 esac

 if [ "$termux" != true ]; then
  # Test for the right location of the "shells" file
  if [ -f /etc/shells ]; then
   shells_file=/etc/shells
  elif [ -f /usr/share/defaults/etc/shells ]; then # Solus OS
   shells_file=/usr/share/defaults/etc/shells
  else
   error "could not find /etc/shells file. Change your default shell manually."
   return
  fi

  # Get the path to the right zsh binary
  # 1. Use the most preceding one based on $PATH, then check that it's in the shells file
  # 2. If that fails, get a zsh path from the shells file, then check it actually exists
  if ! zsh=$(which zsh) || ! grep -qx "$zsh" "$shells_file"; then
   if ! zsh=$(grep '^/.*/zsh$' "$shells_file" | tail -1) || [ ! -f "$zsh" ]; then
    error "no zsh binary found or not present in '$shells_file'"
    error "change your default shell manually."
    return
   fi
  fi
 fi

 # We're going to change the default shell, so back up the current one
 if [ -n "$SHELL" ]; then
  echo $SHELL > ~/.shell.pre-oh-my-zsh
 else
  grep "^$USER:" /etc/passwd | awk -F: '{print $7}' > ~/.shell.pre-oh-my-zsh
 fi

 # Actually change the default shell to zsh
 if ! chsh -s "$zsh"; then
  error "chsh command unsuccessful. Change your default shell manually."
 else
  export SHELL="$zsh"
  echo "${GREEN}Shell successfully changed to '$zsh'.${RESET}"
 fi

 echo
}

main() {
 # Run as unattended if stdin is closed
 if [ ! -t 0 ]; then
  RUNZSH=no
  CHSH=no
 fi

 # Parse arguments
 while [ $# -gt 0 ]; do
  case $1 in
   --unattended) RUNZSH=no; CHSH=no ;;
   --skip-chsh) CHSH=no ;;
  esac
  shift
 done

 setup_color

 if ! command_exists zsh; then
  echo "${YELLOW}Zsh is not installed.${RESET} Please install zsh first."
  exit 1
 fi

 if [ -d "$ZSH" ]; then
  cat <<-EOF
   ${YELLOW}You already have Oh My Zsh installed.${RESET}
   You'll need to remove '$ZSH' if you want to reinstall.
EOF
  exit 1
 fi

 setup_ohmyzsh
 setup_zshrc
 setup_shell

 printf "$GREEN"
 cat <<-'EOF'
           __                                     __
    ____  / /_     ____ ___  __  __   ____  _____/ /_
   / __ \/ __ \   / __ `__ \/ / / /  /_  / / ___/ __ \
  / /_/ / / / /  / / / / / / /_/ /    / /_(__  ) / / /
  \____/_/ /_/  /_/ /_/ /_/\__, /    /___/____/_/ /_/
                          /____/                       ....is now installed!


  Please look over the ~/.zshrc file to select plugins, themes, and options.

  p.s. Follow us on https://twitter.com/ohmyzsh

  p.p.s. Get stickers, shirts, and coffee mugs at https://shop.planetargon.com/collections/oh-my-zsh

EOF
 printf "$RESET"

 if [ $RUNZSH = no ]; then
  echo "${YELLOW}Run zsh to try it out.${RESET}"
  exit
 fi

 exec zsh -l
}

main "$@"
```

安装完成结果如下：

{% asset_img ohmyzsh.png %}

oh-my-zsh安装成功会在根目录下重写`.zshrc`文件，这是zsh的配置文件，可以在其中修改主题以及增加插件等等

```shell
# 比如设置主题为agnoster
ZSH_THEME="agnoster"
```

以及安装命令提示插件`zsh-autosuggestions`，首先在终端里执行以下命令

```shell
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
```

然后修改`.zshrc`文件，修改完记得`source .zshrc`，使配置生效

```shell
plugins=(
    git
    zsh-autosuggestions
)
```

最终效果如下，包括命令提示功能

{% asset_img auto.png %}

## 4. Linuxbrew安装

**注意先安装zsh再安装linuxbrew，否则可能出现brew无法在zsh中使用**

### 4.1 创建非root用户（可选）

安装Linuxbrew参考官网不一定有效，因为Linuxbrew不能在root用户下安装，所以你如果之前登陆的是root用户需要创建一个新的非root用户，通过以下几个指令

```shell
# 创建your_user，需要设置密码
sudo adduser your_user

# 添加your_user到sudo组
sudo adduser your_user sudo
```

然后修改系统中/etc/sudoers文件的方法分配用户权限

```shell
sudo chmod +w /etc/sudoers
sudo vim /etc/sudoers
```

```shell
# User privilege specification
root　ALL=(ALL:ALL) ALL
# 新增your_user
your_user ALL=(ALL:ALL) ALL    
```

将sudoers文件的操作权限还原只读模式

```shell
sudo chmod -w /etc/sudoers
```

最后再登录`your_user`

```shell
su - your_user
```

### 4.2 Linuxbrew

如果已经是非root用户则可以跳过以上创建新用户步骤。

按照Linuxbrew官网的指令安装brew，在这里可能会出现问题，我的华为云服务器这里会报443错误，无法访问`raw.githubusercontent.com`，而腾讯云没有问题，安装成功，需要几分钟。

```shell
sh -c "$(curl -fsSL https://raw.githubusercontent.com/Linuxbrew/install/master/install.sh)"
```

如果出现443错误`Failed to connect to raw.githubusercontent.com port 443: Operation timed out`，则直接将以下内容保存为`brew_install.rb`，然后执行`ruby brew_install.rb`即可，如果系统没有ruby，则执行

```shell
sudo apt install ruby
```

以下内容来自于`https://raw.githubusercontent.com/Linuxbrew/install/master/install`，如有更新，请翻墙访问

```ruby
#!/usr/bin/env ruby
# On macOS, this script installs to /usr/local only.
# On Linux, it installs to /home/linuxbrew/.linuxbrew if you have sudo access
# and ~/.linuxbrew otherwise.
# To install elsewhere (which is unsupported) you can untar
# https://github.com/Homebrew/brew/tarball/master anywhere you like.
# or set the environment variable HOMEBREW_PREFIX.

require "fileutils"

def mac?
  RUBY_PLATFORM[/darwin/]
end

BREW_REPO = "https://github.com/Homebrew/brew".freeze
if mac?
  HOMEBREW_PREFIX = "/usr/local".freeze
  HOMEBREW_REPOSITORY = "/usr/local/Homebrew".freeze
  HOMEBREW_CACHE = "#{ENV["HOME"]}/Library/Caches/Homebrew".freeze
  HOME_CACHE = nil
  CHOWN = "/usr/sbin/chown".freeze
  CHGRP = "/usr/bin/chgrp".freeze
else
  HOMEBREW_PREFIX_DEFAULT = "/home/linuxbrew/.linuxbrew".freeze
  HOME_CACHE = "#{ENV["HOME"]}/.cache".freeze
  HOMEBREW_CACHE = "#{HOME_CACHE}/Homebrew".freeze
  CHOWN = "/bin/chown".freeze
  CHGRP = "/bin/chgrp".freeze
end

# TODO: bump version when new macOS is released
MACOS_LATEST_SUPPORTED = "10.15".freeze
# TODO: bump version when new macOS is released
MACOS_OLDEST_SUPPORTED = "10.13".freeze

# no analytics during installation
ENV["HOMEBREW_NO_ANALYTICS_THIS_RUN"] = "1"
ENV["HOMEBREW_NO_ANALYTICS_MESSAGE_OUTPUT"] = "1"

# get nicer global variables
require "English"

module Tty
  module_function

  def blue
    bold 34
  end

  def red
    bold 31
  end

  def reset
    escape 0
  end

  def bold(code = 39)
    escape "1;#{code}"
  end

  def underline
    escape "4;39"
  end

  def escape(code)
    "\033[#{code}m" if STDOUT.tty?
  end
end

class Array
  def shell_s
    cp = dup
    first = cp.shift
    cp.map { |arg| arg.gsub " ", "\\ " }.unshift(first).join(" ")
  end
end

def ohai(*args)
  puts "#{Tty.blue}==>#{Tty.bold} #{args.shell_s}#{Tty.reset}"
end

def warn(warning)
  puts "#{Tty.red}Warning#{Tty.reset}: #{warning.chomp}"
end

def system(*args)
  abort "Failed during: #{args.shell_s}" unless Kernel.system(*args)
end

def sudo?
  return @have_sudo unless @have_sudo.nil?

  Kernel.system "/usr/bin/sudo", "-l", "mkdir"
  @have_sudo = $CHILD_STATUS.success?
rescue Interrupt
  exit
end

def sudo(*args)
  if sudo?
    args.unshift("-A") unless ENV["SUDO_ASKPASS"].nil?
    ohai "/usr/bin/sudo", *args
    system "/usr/bin/sudo", *args
  else
    ohai *args
    system *args
  end
end

def getc
  system "/bin/stty raw -echo"
  if STDIN.respond_to?(:getbyte)
    STDIN.getbyte
  else
    STDIN.getc
  end
ensure
  system "/bin/stty -raw echo"
end

def wait_for_user
  puts
  puts "Press RETURN to continue or any other key to abort"
  c = getc
  # we test for \r and \n because some stuff does \r instead
  abort unless (c == 13) || (c == 10)
end

class Version
  include Comparable
  attr_reader :parts

  def initialize(str)
    @parts = str.split(".").map(&:to_i)
  end

  def <=>(other)
    parts <=> self.class.new(other).parts
  end

  def to_s
    parts.join(".")
  end
end

def macos_version
  return unless mac?

  @macos_version ||= Version.new(`/usr/bin/sw_vers -productVersion`.chomp[/10\.\d+/])
end

def should_install_command_line_tools?
  return false unless mac?

  if macos_version > "10.13"
    !File.exist?("/Library/Developer/CommandLineTools/usr/bin/git")
  else
    !File.exist?("/Library/Developer/CommandLineTools/usr/bin/git") ||
      !File.exist?("/usr/include/iconv.h")
  end
end

def user_only_chmod?(path)
  return false unless File.directory?(path)

  mode = File.stat(path).mode & 0777
  # u = (mode >> 6) & 07
  # g = (mode >> 3) & 07
  # o = (mode >> 0) & 07
  mode != 0755
end

def chmod?(path)
  File.exist?(path) && !(File.readable?(path) && File.writable?(path) && File.executable?(path))
end

def chown?(path)
  !File.owned?(path)
end

def chgrp?(path)
  !File.grpowned?(path)
end

# return the shell profile file based on users' preference shell
def shell_profile
  case ENV["SHELL"]
  when %r{/bash$} then File.readable?("#{ENV["HOME"]}/.bash_profile") ? "~/.bash_profile" : "~/.profile"
  when %r{/zsh$} then "~/.zprofile"
  else "~/.profile"
  end
end

# USER isn't always set so provide a fall back for the installer and subprocesses.
ENV["USER"] ||= `id -un`.chomp

# Invalidate sudo timestamp before exiting (if it wasn't active before).
Kernel.system "/usr/bin/sudo -n -v 2>/dev/null"
at_exit { Kernel.system "/usr/bin/sudo", "-k" } unless $CHILD_STATUS.success?

# The block form of Dir.chdir fails later if Dir.CWD doesn't exist which I
# guess is fair enough. Also sudo prints a warning message for no good reason
Dir.chdir "/usr"

####################################################################### script
unless mac?
  if File.writable?(HOMEBREW_PREFIX_DEFAULT) || File.writable?("/home/linuxbrew") || File.writable?("/home")
    HOMEBREW_PREFIX = HOMEBREW_PREFIX_DEFAULT.freeze
  else
    sudo_output = `/usr/bin/sudo -n -l mkdir 2>&1`
    if !$CHILD_STATUS.success? && sudo_output == "sudo: a password is required\n"
      ohai "Select the Homebrew installation directory"
      puts "- #{Tty.bold}Enter your password#{Tty.reset} to install to #{Tty.underline}#{HOMEBREW_PREFIX_DEFAULT}#{Tty.reset} (#{Tty.bold}recommended#{Tty.reset})"
      puts "- #{Tty.bold}Press Control-D#{Tty.reset} to install to #{Tty.underline}#{ENV["HOME"]}/.linuxbrew#{Tty.reset}"
      puts "- #{Tty.bold}Press Control-C#{Tty.reset} to cancel installation"
    end
    if sudo?
      HOMEBREW_PREFIX = HOMEBREW_PREFIX_DEFAULT.freeze
    else
      HOMEBREW_PREFIX = "#{ENV["HOME"]}/.linuxbrew".freeze
    end
  end
  HOMEBREW_REPOSITORY = "#{HOMEBREW_PREFIX}/Homebrew".freeze
end

if mac? && macos_version < "10.7"
  abort <<-EOABORT
Your Mac OS X version is too old. See:
  #{Tty.underline}https://github.com/mistydemeo/tigerbrew#{Tty.reset}"
  EOABORT
elsif mac? && macos_version < "10.9"
  abort "Your OS X version is too old"
elsif Process.uid.zero?
  abort "Don't run this as root!"
elsif mac? && !`dsmemberutil checkmembership -U "#{ENV["USER"]}" -G admin`.include?("user is a member")
  abort "This script requires the user #{ENV["USER"]} to be an Administrator."
elsif File.directory?(HOMEBREW_PREFIX) && (!File.executable? HOMEBREW_PREFIX)
  abort <<-EOABORT
The Homebrew prefix, #{HOMEBREW_PREFIX}, exists but is not searchable. If this is
not intentional, please restore the default permissions and try running the
installer again:
    sudo chmod 775 #{HOMEBREW_PREFIX}
  EOABORT
# TODO: bump version when new macOS is released
elsif mac? && (macos_version > MACOS_LATEST_SUPPORTED || macos_version < MACOS_OLDEST_SUPPORTED)
  who = "We"
  if macos_version > MACOS_LATEST_SUPPORTED
    what = "pre-release version"
  elsif macos_version < MACOS_OLDEST_SUPPORTED
    who << " (and Apple)"
    what = "old version"
  else
    return
  end
  ohai "You are using macOS #{macos_version.parts.join(".")}."
  ohai "#{who} do not provide support for this #{what}."

  puts <<-EOS
This installation may not succeed.
After installation, you will encounter build failures with some formulae.
Please create pull requests instead of asking for help on Homebrew's GitHub,
Discourse, Twitter or IRC. You are responsible for resolving any issues you
experience while you are running this #{what}.

  EOS
end

ohai "This script will install:"
puts "#{HOMEBREW_PREFIX}/bin/brew"
puts "#{HOMEBREW_PREFIX}/share/doc/homebrew"
puts "#{HOMEBREW_PREFIX}/share/man/man1/brew.1"
puts "#{HOMEBREW_PREFIX}/share/zsh/site-functions/_brew"
puts "#{HOMEBREW_PREFIX}/etc/bash_completion.d/brew"
puts "#{HOMEBREW_CACHE}/"
puts HOMEBREW_REPOSITORY.to_s

# Keep relatively in sync with
# https://github.com/Homebrew/brew/blob/master/Library/Homebrew/keg.rb
group_chmods = %w[bin etc include lib sbin share opt var
                  Frameworks
                  etc/bash_completion.d lib/pkgconfig
                  share/aclocal share/doc share/info share/locale share/man
                  share/man/man1 share/man/man2 share/man/man3 share/man/man4
                  share/man/man5 share/man/man6 share/man/man7 share/man/man8
                  var/log var/homebrew var/homebrew/linked
                  bin/brew]
               .map { |d| File.join(HOMEBREW_PREFIX, d) }
               .select { |d| chmod?(d) }
# zsh refuses to read from these directories if group writable
zsh_dirs = %w[share/zsh share/zsh/site-functions]
           .map { |d| File.join(HOMEBREW_PREFIX, d) }
mkdirs = %w[bin etc include lib sbin share var opt
            share/zsh share/zsh/site-functions
            var/homebrew var/homebrew/linked
            Cellar Caskroom Homebrew Frameworks]
         .map { |d| File.join(HOMEBREW_PREFIX, d) }
         .reject { |d| File.directory?(d) }

user_chmods = zsh_dirs.select { |d| user_only_chmod?(d) }
chmods = group_chmods + user_chmods
chowns = chmods.select { |d| chown?(d) }
chgrps = chmods.select { |d| chgrp?(d) }

group = `id -gn`.chomp
abort "error: id -gn: failed" unless $CHILD_STATUS.success? && !group.empty?

unless group_chmods.empty?
  ohai "The following existing directories will be made group writable:"
  puts(*group_chmods)
end
unless user_chmods.empty?
  ohai "The following existing directories will be made writable by user only:"
  puts(*user_chmods)
end
unless chowns.empty?
  ohai "The following existing directories will have their owner set to #{Tty.underline}#{ENV["USER"]}#{Tty.reset}:"
  puts(*chowns)
end
unless chgrps.empty?
  ohai "The following existing directories will have their group set to #{Tty.underline}#{group}#{Tty.reset}:"
  puts(*chgrps)
end
unless mkdirs.empty?
  ohai "The following new directories will be created:"
  puts(*mkdirs)
end
if should_install_command_line_tools?
  ohai "The Xcode Command Line Tools will be installed."
end

wait_for_user if STDIN.tty? && !ENV["CI"]

if File.directory? HOMEBREW_PREFIX
  sudo "/bin/chmod", "u+rwx", *chmods unless chmods.empty?
  sudo "/bin/chmod", "g+rwx", *group_chmods unless group_chmods.empty?
  sudo "/bin/chmod", "755", *user_chmods unless user_chmods.empty?
  sudo CHOWN, ENV["USER"], *chowns unless chowns.empty?
  sudo CHGRP, group, *chgrps unless chgrps.empty?
else
  sudo "/bin/mkdir", "-p", HOMEBREW_PREFIX
  sudo CHOWN, "#{ENV["USER"]}:#{group}", HOMEBREW_PREFIX
end

unless mkdirs.empty?
  sudo "/bin/mkdir", "-p", *mkdirs
  sudo "/bin/chmod", "g+rwx", *mkdirs
  sudo "/bin/chmod", "755", *zsh_dirs
  sudo CHOWN, ENV["USER"], *mkdirs
  sudo CHGRP, group, *mkdirs
end

sudo "/bin/mkdir", "-p", HOMEBREW_CACHE unless File.directory? HOMEBREW_CACHE
sudo "/bin/chmod", "g+rwx", HOMEBREW_CACHE if chmod? HOMEBREW_CACHE
sudo CHOWN, ENV["USER"], HOMEBREW_CACHE if chown? HOMEBREW_CACHE
sudo CHGRP, group, HOMEBREW_CACHE if chgrp? HOMEBREW_CACHE
if HOME_CACHE
  sudo CHOWN, ENV["USER"], HOME_CACHE if chown? HOME_CACHE
  sudo CHGRP, group, HOME_CACHE if chgrp? HOME_CACHE
end
FileUtils.touch "#{HOMEBREW_CACHE}/.cleaned" if File.directory? HOMEBREW_CACHE

if should_install_command_line_tools? && macos_version >= "10.13"
  ohai "Searching online for the Command Line Tools"
  # This temporary file prompts the 'softwareupdate' utility to list the Command Line Tools
  clt_placeholder = "/tmp/.com.apple.dt.CommandLineTools.installondemand.in-progress"
  sudo "/usr/bin/touch", clt_placeholder

  clt_label_command = "/usr/sbin/softwareupdate -l | " \
                      "grep -B 1 -E 'Command Line Tools' | " \
                      "awk -F'*' '/^ *\\*/ {print $2}' | " \
                      "sed -e 's/^ *Label: //' -e 's/^ *//' | " \
                      "sort -V | " \
                      "tail -n1"
  clt_label = `#{clt_label_command}`.chomp

  unless clt_label.empty?
    ohai "Installing #{clt_label}"
    sudo "/usr/sbin/softwareupdate", "-i", clt_label
    sudo "/bin/rm", "-f", clt_placeholder
    sudo "/usr/bin/xcode-select", "--switch", "/Library/Developer/CommandLineTools"
  end
end

# Headless install may have failed, so fallback to original 'xcode-select' method
if should_install_command_line_tools? && STDIN.tty?
  ohai "Installing the Command Line Tools (expect a GUI popup):"
  sudo "/usr/bin/xcode-select", "--install"
  puts "Press any key when the installation has completed."
  getc
  sudo "/usr/bin/xcode-select", "--switch", "/Library/Developer/CommandLineTools"
end

abort <<-EOABORT if mac? && `/usr/bin/xcrun clang 2>&1` =~ /license/ && !$CHILD_STATUS.success?
You have not agreed to the Xcode license.
Before running the installer again please agree to the license by opening
Xcode.app or running:
    sudo xcodebuild -license
EOABORT

ohai "Downloading and installing Homebrew..."
Dir.chdir HOMEBREW_REPOSITORY do
  # we do it in four steps to avoid merge errors when reinstalling
  system "git", "init", "-q"

  # "git remote add" will fail if the remote is defined in the global config
  system "git", "config", "remote.origin.url", BREW_REPO
  system "git", "config", "remote.origin.fetch", "+refs/heads/*:refs/remotes/origin/*"

  # ensure we don't munge line endings on checkout
  system "git", "config", "core.autocrlf", "false"

  system "git", "fetch", "origin", "master:refs/remotes/origin/master",
         "--tags", "--force"

  system "git", "reset", "--hard", "origin/master"

  system "ln", "-sf", "#{HOMEBREW_REPOSITORY}/bin/brew", "#{HOMEBREW_PREFIX}/bin/brew"

  system "#{HOMEBREW_PREFIX}/bin/brew", "update", "--force"
end

ohai "Installation successful!"
puts

# Use the shell's audible bell.
print "\a"

# Use an extra newline and bold to avoid this being missed.
ohai "Homebrew has enabled anonymous aggregate formulae and cask analytics."
puts <<-EOS
#{Tty.bold}Read the analytics documentation (and how to opt-out) here:
  #{Tty.underline}https://docs.brew.sh/Analytics#{Tty.reset}

EOS

ohai "Homebrew is run entirely by unpaid volunteers. Please consider donating:"
puts <<-EOS
  #{Tty.underline}https://github.com/Homebrew/brew#donations#{Tty.reset}
EOS

Dir.chdir HOMEBREW_REPOSITORY do
  system "git", "config", "--replace-all", "homebrew.analyticsmessage", "true"
  system "git", "config", "--replace-all", "homebrew.caskanalyticsmessage", "true"
end

ohai "Next steps:"

unless mac?
  puts <<-EOS
- Install the Homebrew dependencies if you have sudo access:
  #{Tty.bold}Debian, Ubuntu, etc.#{Tty.reset}
    sudo apt-get install build-essential
  #{Tty.bold}Fedora, Red Hat, CentOS, etc.#{Tty.reset}
    sudo yum groupinstall 'Development Tools'
  See #{Tty.underline}https://docs.brew.sh/linux#{Tty.reset} for more information.
- Configure Homebrew in your #{Tty.underline}#{shell_profile}#{Tty.reset} by running
    echo 'eval $(#{HOMEBREW_PREFIX}/bin/brew shellenv)' >>#{shell_profile}
- Add Homebrew to your #{Tty.bold}PATH#{Tty.reset}
    eval $(#{HOMEBREW_PREFIX}/bin/brew shellenv)
- We recommend that you install GCC by running:
    brew install gcc
  EOS
end

puts "- Run `brew help` to get started"
puts "- Further documentation: "
puts "    #{Tty.underline}https://docs.brew.sh#{Tty.reset}"

warn "#{HOMEBREW_PREFIX}/bin is not in your PATH." unless ENV["PATH"].split(":").include? "#{HOMEBREW_PREFIX}/bin"
```

安装成功结果如下，这里需要按照提示依次执行

```shell
sudo apt-get install build-essential

echo 'eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv)' >>~/.zprofile

eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv)

brew install gcc
```

就可以正常使用brew了，最后一步安装gcc可能会非常耗时（腾讯云网络速度非常奇葩，有时快有时慢，最后从源码编译gcc花了两个小时）。

```shell
==> Installation successful!

==> Homebrew has enabled anonymous aggregate formulae and cask analytics.
Read the analytics documentation (and how to opt-out) here:
  https://docs.brew.sh/Analytics

==> Homebrew is run entirely by unpaid volunteers. Please consider donating:
  https://github.com/Homebrew/brew#donations
==> Next steps:
- Install the Homebrew dependencies if you have sudo access:
  Debian, Ubuntu, etc.
    sudo apt-get install build-essential
  Fedora, Red Hat, CentOS, etc.
    sudo yum groupinstall 'Development Tools'
  See https://docs.brew.sh/linux for more information.
- Configure Homebrew in your ~/.zprofile by running
    echo 'eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv)' >>~/.zprofile
- Add Homebrew to your PATH
    eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv)
- We recommend that you install GCC by running:
    brew install gcc
- Run `brew help` to get started
- Further documentation: 
    https://docs.brew.sh
Warning: /home/linuxbrew/.linuxbrew/bin is not in your PATH.
```

若brew正确安装，则可以通过`brew -v`查看brew版本信息

```shell
Homebrew 2.2.2
Homebrew/linuxbrew-core (git revision 906b; last commit 2020-01-11)
```

### 4.3 腾讯云下载问题

在安装linuxbrew时，我发现腾讯云服务器在下载[`portable-ruby-2.6.3.x86_64_linux.bottle.tar.gz`](https://linuxbrew.bintray.com/bottles-portable-ruby/portable-ruby-2.6.3.x86_64_linux.bottle.tar.gz)这个包的时候速度非常慢并且导致超时（华为云没有问题），结果brew安装失败，如果失败了可以先将linuxbrew删除（执行`sudo rm -rf /home/linuxbrew`，具体看你的linuxbrew安装目录），再通过其他方式预先把`portable-ruby-2.6.3.x86_64_linux.bottle.tar.gz`下载下来并且放到`~/.cache/Homebrew/`目录下，这样重新安装linuxbrew时就会直接从`.cache`中解压安装了。

### 4.4 brew更换源（**Ubuntu不要使用，macOS可以使用**）

brew下载某些软件时会因为网络原因非常慢，甚至导致安装失败的问题，所以可以使用国内源，比如[清华大学开源软件镜像站](https://mirror.tuna.tsinghua.edu.cn/help/homebrew/)以及[中科大镜像源](https://lug.ustc.edu.cn/wiki/mirrors/help/brew.git)，可以更换4个位置的源，分别是`brew/homebrew-core/homebrew-cask/homebrew-bottles`，前三个可以修改本地仓库的信息，最后一个需要修改`.zshrc`。

```shell
# brew自己的仓库
git -C "$(brew --repo)" remote set-url origin https://mirrors.ustc.edu.cn/brew.git
# brew可以安装的软件名称的仓库
git -C "$(brew --repo homebrew/core)" remote set-url origin https://mirrors.ustc.edu.cn/homebrew-core.git
# brew可以安装的GUI软件的仓库，如果提示没有cask，可以先执行brew cask
git -C "$(brew --repo homebrew/cask)" remote set-url origin https://mirrors.ustc.edu.cn/homebrew-cask.git

# 更新brew自己
brew update

# 更新brew安装的软件
brew upgrade

# 清楚无效连接以及本地下载的安装文件缓存
brew cleanup
```

```shell
# brew可以安装的软件的仓库
echo 'export HOMEBREW_BOTTLE_DOMAIN=https://mirrors.ustc.edu.cn/homebrew-bottles' >> ~/.zshrc
source ~/.zshrc
```

### 4.5 卸载linuxbrew

执行以下命令，同理如果出现443问题，翻墙查看并保存，然后直接运行

```shell
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Linuxbrew/install/master/uninstall)"
```


