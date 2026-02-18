import os

file_path = 'requirements.txt'
target = 'numpy==2.3.5'
replacement = 'numpy<2'

try:
    print(f"Reading {file_path} with utf-16")
    with open(file_path, 'r', encoding='utf-16') as f:
        content = f.read()
except Exception as e:
    print(f"Failed with utf-16: {e}")
    try:
        print(f"Reading {file_path} with utf-8")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Failed with utf-8: {e}")
        exit(1)

if target in content:
    new_content = content.replace(target, replacement)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"Successfully updated {file_path} to use {replacement}")
else:
    print(f"'{target}' not found in {file_path}")
    # Check if numpy is there at all
    if 'numpy' in content:
        print("Found numpy lines:")
        for line in content.splitlines():
            if 'numpy' in line:
                print(line)
