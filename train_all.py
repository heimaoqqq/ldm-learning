#!/usr/bin/env python3
"""
DiffGait 完整训练脚本
两阶段训练：VQ-VAE → CLDM
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_header(text):
    """打印格式化的标题"""
    print("\n" + "=" * 80)
    print(f"   {text}")
    print("=" * 80)

def run_command(cmd, cwd=None):
    """运行命令并实时显示输出"""
    print(f"执行命令: {cmd}")
    if cwd:
        print(f"工作目录: {cwd}")
    
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        cwd=cwd,
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # 实时输出
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    
    return_code = process.poll()
    if return_code != 0:
        print(f"❌ 命令执行失败，返回码: {return_code}")
        return False
    else:
        print("✅ 命令执行成功")
        return True

def check_data_path():
    """检查数据集路径"""
    data_paths = [
        "/kaggle/input/dataset/dataset",
        "dataset",
        "../dataset"
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            print(f"✅ 找到数据集路径: {path}")
            return path
    
    print("❌ 未找到数据集，请确认数据集路径")
    return None

def main():
    print_header("DiffGait 两阶段训练开始")
    
    # 检查当前目录
    current_dir = os.getcwd()
    print(f"当前工作目录: {current_dir}")
    
    # 检查数据集
    data_path = check_data_path()
    if not data_path:
        print("请将数据集放在正确位置或修改配置文件中的路径")
        return
    
    start_time = time.time()
    
    # 第一阶段：VQ-VAE训练
    print_header("第一阶段：训练VQ-VAE模型")
    
    vae_dir = "VAE"
    if not os.path.exists(vae_dir):
        print(f"❌ 找不到VAE目录: {vae_dir}")
        return
    
    print("开始VQ-VAE训练...")
    if not run_command("python train_adv_vqvae.py", cwd=vae_dir):
        print("❌ VQ-VAE训练失败")
        return
    
    # 检查VQ-VAE权重文件
    vqvae_weight_path = os.path.join(vae_dir, "ae_debug_models", "adv_vqvae_best_loss.pth")
    if not os.path.exists(vqvae_weight_path):
        print(f"❌ VQ-VAE权重文件不存在: {vqvae_weight_path}")
        return
    
    print(f"✅ VQ-VAE训练完成，权重文件: {vqvae_weight_path}")
    
    # VQ-VAE可视化
    print("生成VQ-VAE可视化结果...")
    run_command("python visualize_adv_vqvae.py", cwd=vae_dir)
    
    # 第二阶段：CLDM训练
    print_header("第二阶段：训练CLDM模型")
    
    ldm_dir = "LDM"
    if not os.path.exists(ldm_dir):
        print(f"❌ 找不到LDM目录: {ldm_dir}")
        return
    
    print("开始CLDM训练...")
    if not run_command("python train.py", cwd=ldm_dir):
        print("❌ CLDM训练失败")
        return
    
    print("✅ CLDM训练完成")
    
    # CLDM可视化
    print("生成CLDM可视化结果...")
    run_command("python visualize.py", cwd=ldm_dir)
    
    # 训练完成
    end_time = time.time()
    total_time = end_time - start_time
    
    print_header("训练完成总结")
    print(f"✅ 两阶段训练全部完成！")
    print(f"⏱️  总训练时间: {total_time/3600:.2f} 小时")
    print(f"📁 VQ-VAE权重: {vqvae_weight_path}")
    print(f"📁 CLDM权重: {os.path.join(ldm_dir, 'models')}")
    print(f"🖼️  可视化结果:")
    print(f"   - VQ-VAE: {os.path.join(vae_dir, 'ae_debug_models', 'samples')}")
    print(f"   - CLDM: {os.path.join(ldm_dir, 'visualizations')}")
    
    print("\n🎉 DiffGait训练成功完成！")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc() 