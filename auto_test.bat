ECHO OFF
ECHO ************************************************************************************************
ECHO Win batch script for running FL with varying lag_tolerance automatically
ECHO This script is supposed to run in the dir where FL.launcher.py locates
ECHO ************************************************************************************************

SET /P crash_prob=specify crash prob: 
ECHO preparing to test lag_t = 1,2,3,4,5 with crash prob = %crash_prob%

FOR %%L IN (1,2,3,4,5) DO fed\python FL_launcher.py %crash_prob% %%L

:: finish
ECHO finished...
PAUSE