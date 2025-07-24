#!/usr/bin/env python3
"""
Matplotlib 한글 폰트 설정 스크립트
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

def setup_korean_font():
    # 시스템에서 사용 가능한 한글 폰트 찾기
    font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    korean_fonts = []
    
    # 일반적인 한글 폰트 이름들
    korean_font_names = ['NanumGothic', 'NanumBarunGothic', 'Malgun Gothic', 
                        'Dotum', 'Batang', 'Gulim', 'DejaVu Sans']
    
    for font_path in font_list:
        try:
            font_prop = fm.FontProperties(fname=font_path)
            font_name = font_prop.get_name()
            for kf in korean_font_names:
                if kf.lower() in font_name.lower():
                    korean_fonts.append((font_name, font_path))
                    break
        except:
            continue
    
    if korean_fonts:
        print("발견된 한글 폰트:")
        for name, path in korean_fonts:
            print(f"  - {name}: {path}")
        
        # 첫 번째 한글 폰트 사용
        font_name, font_path = korean_fonts[0]
        plt.rcParams['font.family'] = font_name
        plt.rcParams['axes.unicode_minus'] = False
        
        print(f"\n'{font_name}' 폰트를 기본으로 설정했습니다.")
        
        # matplotlibrc 파일 생성
        config_dir = os.path.expanduser('~/.config/matplotlib')
        os.makedirs(config_dir, exist_ok=True)
        
        rc_file = os.path.join(config_dir, 'matplotlibrc')
        with open(rc_file, 'w') as f:
            f.write(f"font.family: {font_name}\n")
            f.write("axes.unicode_minus: False\n")
        
        print(f"설정이 {rc_file}에 저장되었습니다.")
    else:
        print("한글 폰트를 찾을 수 없습니다.")
        print("DejaVu Sans를 사용합니다 (한글 미지원).")
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    setup_korean_font()