# Fresco Reconstruction

## To Run the fresco reconstruction for all fragment 
```bash
python3 src/fresco_reconstruction.py --fragments_dir_path=all_fragment_directory_path --fresco_path=fresco_image_path
```

exemple : 
```bash
python3 src/fresco_reconstruction.py --fragments_dir_path="src/images/frag_eroded/" --fresco_path="src/images/image.jpg"
```

## To Run the fresco reconstruction for one fragment 
```bash
python3 src/fresco_reconstruction.py --fragments_dir_path="src/images/frag_eroded/" --fresco_path="src/images/image.jpg" --all=False --index_fragment=fragment_number
```

exemple pour le fragment 92: 
```bash
python3 src/fresco_reconstruction.py --fragments_dir_path="src/images/frag_eroded/" --fresco_path="src/images/image.jpg" --all=False --index_fragment=92
```