import os
import re
import subprocess

chapters = ['04', '05', '06', '07', '08', '09']

def get_frozen_deps(chapter):
    env_dir = f"chapter_{chapter}/.venv_ch{chapter}"
    if not os.path.isdir(env_dir):
        print(f"Environment for chapter {chapter} not found.")
        return {}
    
    cmd = ["uv", "pip", "freeze", "--python", env_dir]
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Error running uv pip freeze for chapter {chapter}: {e}")
        return {}
    
    deps = {}
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '==' in line:
            pkg, ver = line.split('==', 1)
            deps[pkg.lower()] = ver
        elif '@' in line:
            pkg, _ = line.split('@', 1)
            pkg = pkg.strip()
            # We can't really pin git/path installs easily, but we record them.
    return deps

def process_whole_content(chapter, deps):
    script_path = f"chapter_{chapter}/install_ch{chapter}.sh"
    if not os.path.isfile(script_path):
        print(f"Script {script_path} not found.")
        return
        
    with open(script_path, 'r') as f:
        content = f.read()
        
    # Find all occurrences of uv pip install and following arguments.
    tokens = re.split(r'(\s+|\|\||&&|;)', content)
    
    out_tokens = []
    in_install = False
    
    i = 0
    while i < len(tokens):
        t = tokens[i]
        
        if not t.strip():
            out_tokens.append(t)
            if '\n' in t:
                last_non_ws = ""
                for prev in reversed(out_tokens[:-1]):
                    if prev.strip():
                        last_non_ws = prev
                        break
                if not last_non_ws.endswith('\\'):
                    in_install = False
            i += 1
            continue
            
        if t == 'uv' and i+4 < len(tokens) and tokens[i+2] == 'pip' and tokens[i+4] == 'install':
            in_install = True
            out_tokens.extend([tokens[i], tokens[i+1], tokens[i+2], tokens[i+3], tokens[i+4]])
            i += 5
            continue
            
        if t in ('||', '&&', ';'):
            in_install = False
            out_tokens.append(t)
            i += 1
            continue
            
        if in_install and not t.startswith('-') and not t.startswith('git+'):
            clean_t = t.strip('"\'\\')
            m = re.match(r'^([A-Za-z0-9_\-]+)(?:[><=].*)?$', clean_t)
            if m:
                base_name = m.group(1)
                lower_name = base_name.lower().replace('_', '-')
                alt_name = lower_name.replace('-', '_')
                ver = None
                if lower_name in deps:
                    ver = deps[lower_name]
                elif alt_name in deps:
                    ver = deps[alt_name]
                    
                if ver:
                    new_tok = f'"{base_name}=={ver}"'
                    if t.endswith('\\'):
                        new_tok += '\\'
                    out_tokens.append(new_tok)
                    i += 1
                    continue
                    
        out_tokens.append(t)
        i += 1
        
    with open(script_path, 'w') as f:
        f.write("".join(out_tokens))
    print(f"Updated {script_path}")


if __name__ == '__main__':
    for ch in chapters:
        deps = get_frozen_deps(ch)
        if deps:
            process_whole_content(ch, deps)
