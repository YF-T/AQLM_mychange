#!/bin/bash

# 远程服务器信息
REMOTE_USER="tanyf"
REMOTE_HOST="123.127.250.154"
REMOTE_PORT="9103"
REMOTE_DIR="/home/tanyf/AQLM"
PASSWORD="tanyf@1_326"

# 本地目录
LOCAL_DIR="."

# 全局忽略列表
IGNORE_LIST=(
    '.git/'
    '__pycache__/'
)

# 同步函数（增加命令打印）
sync_files() {
    local src=$1
    local dest=$2
    local rsync_exclude_opts=()
    for pattern in "${IGNORE_LIST[@]}"; do
        rsync_exclude_opts+=("--exclude=$pattern")
    done

    # 构建命令数组
    local cmd=(
        sshpass -p "$PASSWORD"
        rsync -avz --update
        "${rsync_exclude_opts[@]}"
        -e "ssh -p $REMOTE_PORT"
        --progress
        "$src"
        "$dest"
    )

    # 打印实际将要执行的命令
    echo "实际执行命令："
    printf "%q " "${cmd[@]}"
    echo

    # 执行命令
    "${cmd[@]}"
}

# 从远程同步到本地
echo "正在从远程同步文件到本地 (目录: $REMOTE_DIR)..."
sync_files "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/" "$LOCAL_DIR/"

# 从本地同步到远程
echo "正在从本地同步文件到远程 (目录: $REMOTE_DIR)..."
sync_files "$LOCAL_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"

echo "同步完成！"
