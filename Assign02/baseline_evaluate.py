import pix2pix.test as p2p

def main():
    p2p.main(["--dataroot", "./Assign02/Facades", "--name", "facades_pix2pix", 
              "--model", "pix2pix", "--direction", "BtoA"])
    
if __name__ == "__main__":
    main()
    
    