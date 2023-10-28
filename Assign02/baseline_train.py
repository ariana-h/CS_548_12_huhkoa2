import pix2pix.train as p2p

def main():
    p2p.main(["--dataroot", "./Assign02/Facades", "--name", "facades_pix2pix", "--model", "pix2pix", "--direction", "BtoA", "--display_id", "-1"])
 
if __name__ == "__main__":
    main()
    
    