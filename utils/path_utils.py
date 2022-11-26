
from pathlib import Path
import regex

def sorted_file_paths(imgs_dir, img_pattern, sort_key_pattern=None, by_int = True, to_str = False):
    '''
    Args: 
        img_dir: (str) directory storing images.
        img_pattern: (str) glob pattern, NOT regex.
        sort_key_pattern: (str) regex pattern used to sort images.
    
    Return: 
        (list[pathlib.Path]) List of image paths.
    '''
    if not type(imgs_dir) == 'pathlib.PosixPath':
        imgs_dir = Path(imgs_dir)
    if sort_key_pattern:
        pattern = regex.compile(sort_key_pattern)
    if by_int == True:
        return [str(img_path) if to_str else img_path for img_path in sorted(imgs_dir.glob(img_pattern), key=lambda x: int(''.join(filter(str.isdigit, x.stem))))]
    else:
        return [str(img_path) if to_str else img_path for img_path in sorted(imgs_dir.glob(img_pattern), key=lambda x: int(regex.search(pattern, str(x)).group(0)))]

