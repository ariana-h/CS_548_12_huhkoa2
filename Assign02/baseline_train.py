#using https://github.com/saba99/pix2pix-Facades

import pix2pix.train as p2p

def main():
    p2p.main(["--dataroot", "./Assign02/Facades", "--name", "facades_pix2pix", 
              "--model", "pix2pix", "--direction", "BtoA", "--display_id", "-1", "--verbose"])
 
if __name__ == "__main__":
    main()
    
