from time import sleep
import socket

BUFSIZE = 500

# Echo client program

HOST = 'localhost'    # The remote host
PORT = 1234         # The same port as used by the server

def main():

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))

    f = open('Data/sampledata', 'r')
    lines = f.readlines()
    l  = len(lines)
    f.close()
    
    x = 0
    while True:
        if x + BUFSIZE >= l:
            x = 0

        else:
            buf = lines[x:x+BUFSIZE]
            bufs = "".join(buf) 
            s.send(bufs)
            print bufs
            
            #bufs = "".join(buf) + "\n"
            #print "\n" 
            sleep(0.5)
        x = x + BUFSIZE

    s.close()

if __name__ == '__main__':
    main()         
