#!/bin/bash

# results 디렉토리로 이동
cd results || { echo "❌ results directory not found"; exit 1; }

# sqrt_NF를 포함하는 모든 디렉토리 찾기
for dir in *sqrt_NF-*; do
    # 디렉토리가 실제로 존재하는지 확인
    if [ -d "$dir" ]; then
        # sqrt_NF를 sqrt_NF_2p로 변경
        new_dir="${dir/sqrt_NF-/sqrt_NF_2p-}"
        
        # 이미 변경된 이름이 존재하는지 확인
        if [ -d "$new_dir" ]; then
            echo "⚠️  $new_dir already exists, skipping $dir"
        else
            echo "Renaming: $dir -> $new_dir"
            mv "$dir" "$new_dir"
        fi
    fi
done

echo "✅ Done!"