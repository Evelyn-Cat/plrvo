bash run_7.PG.cv.sh Gaussian
wait

for idx in 2 3 4 5 6 7
do
    bash run_7.PG.cv.sh PLRVO $idx
    wait
done
