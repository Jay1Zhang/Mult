echo FOLD7
python main.py --verbose --fd 7 >> log.txt
cp -r new_trained_models/fold7/ pre_trained_models/
python main.py --test_mode --fd 7 >> log.txt
echo FOLD8
python main.py --verbose --fd 8 >> log.txt
cp -r new_trained_models/fold8/ pre_trained_models/
python main.py --test_mode --fd 8 >> log.txt
echo FOLD9
python main.py --verbose --fd 9 >> log.txt
cp -r new_trained_models/fold9/ pre_trained_models/
python main.py --test_mode --fd 9 >> log.txt
