cpp: 
	clear && cd build && make && clear && ./bin/fresco-reconstruction ../src/images/image.jpg

py: 
	clear && python3 src/fresco_reconstruction.py

md: 
	clear && eval "$(/usr/libexec/path_helper)" && pandoc docs/rapport.md --pdf-engine=xelatex -o docs/rapport.pdf && open docs/rapport.pdf