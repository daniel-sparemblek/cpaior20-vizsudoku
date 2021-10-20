 #!/bin/bash
 echo "downloading sudoku datasets.."
 wget -cq powei.tw/sudoku.zip && unzip -qq sudoku.zip
 if [ ! -d "data" ]
 then
	 mkdir data
 fi
 mv sudoku/* ./data
 rm -r sudoku
 rm sudoku.zip
 echo "done"
