rclone sync data/ :sftp,host=192.168.2.162,user=nathan,ask_password=true:/home/nathan/APS360/data/ --transfers 32 --checkers 64 --sftp-disable-hashcheck --sftp-chunk-size 255k -P
