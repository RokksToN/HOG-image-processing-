float detect(PImage img1,PImage img2,int b[],int n){
  
  img1.loadPixels();
  img2.loadPixels();
  
  float[][] sobel1 = {{  -1,  0,  1  }, 	//sobel filters used for calculating gradient vectors with kernel 3x3
                    { -2, 0, 2 }, 
                    {  -1,  0,  1 }};
  float[][] sobel2 = {{  -1,  -2,  -1  }, 
                    { 0, 0, 0 }, 
                    {  1,  2,  1 }};
                    
  
  float[] c1=new float[(b[0]*b[1]*n)]; 		//merging of all feature vectors, size is number of blocks * chaincode directions(n)
  float[] c2=new float[(b[0]*b[1]*n)];		//when segmenting image on blocks, b[0] is number of rows and b[1] is number of columns
  float powc1=0;		//sum of all elements squared individually
  float powc2=0;
  int block=0; 		//next free block
  
  float movey=ceil((1.0*img1.height/b[0]));		//block size
  float movex=ceil((1.0*img1.width/b[1]));
  
  for(int i=1;i<img1.height-1;i=i+(int)movey) 	//moving of blocks
    for(int j=1;j<img1.width-1;j=j+(int)movex){
      
      float[] b1= new float[n];		//used for summing up feature vectors in this block
      
      for (int y = i; (y < i+movey) && (y < img1.height-1); y++) 	//moving by pixels in each block
        for (int x = j; (x < j+movex) && (x < img1.width-1); x++) {
		float gx=0,gy=0;	//magnitude of gradient vector |gx|+|gy|
          float teta=0;		//gradient orientation
          //float[] a=new float[n]; 
          for (int ky = -1; ky <= 1; ky++) 		//application of sobel filters
            for (int kx = -1; kx <= 1; kx++) {
             int index = (y + ky) * img1.width + (x + kx);
             float r = brightness(img1.pixels[index]);
             gx += sobel1[ky+1][kx+1]*r;
             gy += sobel2[ky+1][kx+1]*r;
            }
        
      if(gy>=0)teta=atan2(gy,gx);		//atan2 returns gradient orientation from -pi to pi, so if gy is below 0 we need to add 2 pi to make is positive
        else teta=atan2(gy,gx)+2*PI;	
        
      //a[(int)(teta*n/(2*PI))]=abs(gx)+abs(gy);  feature vector of pixel
        
      b1[(int)(teta*n/(2*PI))]+=abs(gx)+abs(gy);  //sum of feature vectors in block(each feature vector is projected in the elementary vector on his right(index))
      
           
  }
      for(int bi=0;bi<n;bi++){		//placing sums in array
        c1[block*n+bi]=b1[bi];
        powc1+=b1[bi]*b1[bi];
      }
      block++;	//moving to next block
  }
  
  block=0; //new picture
  
  movey=ceil((1.0*img2.height/b[0]));	//block size
  movex=ceil((1.0*img2.width/b[1]));
  
  for(int i=1;i<img2.height-1;i=i+(int)movey) 	//moving of blocks
    for(int j=1;j<img2.width-1;j=j+(int)movex){
      
      float[] b1= new float[n];
      
      for (int y = i; (y < i+movey) && (y < img2.height-1); y++) //moving by pixels in each block
        for (int x = j; (x < j+movex) && (x < img2.width-1); x++) {
          float gx=0,gy=0;
          float teta=0;
           
          for (int ky = -1; ky <= 1; ky++) 
            for (int kx = -1; kx <= 1; kx++) {
             int index = (y + ky) * img2.width + (x + kx);
             float r = brightness(img2.pixels[index]);
             gx += sobel1[ky+1][kx+1]*r;
             gy += sobel2[ky+1][kx+1]*r;
            }
        
      if(gy>=0)teta=atan2(gy,gx);
        else teta=atan2(gy,gx)+2*PI;
        
        
      b1[(int)(teta*n/(2*PI))]+=abs(gx)+abs(gy);  //sum of all feature vectors representing pixels in the observed block
      
           
  }
      for(int bi=0;bi<n;bi++){
        c2[block*n+bi]=b1[bi];
        powc2+=b1[bi]*b1[bi];
      }
      block++;
  }
  
  float skalar=0,pownorm1=0,pownorm2=0;
  
  for(int i=0;i<block*n;i++){ 
      c1[i]=c1[i]/powc1;      // normalisation, each element(feature vector) is divided by sum of all elements squared individually
      c2[i]=c2[i]/powc2;	  // normalisation is done so we can compare the pictures	
      skalar+=c1[i]*c2[i];      //scalar product
      pownorm1+=c1[i]*c1[i];      //sum of all elements squared individually after normalisation
      pownorm2+=c2[i]*c2[i];
      }
      
  return skalar/(sqrt(pownorm1)*sqrt(pownorm2));	//cosine similarity, value will be in range [0,1] where greater value indicates higher similarity

}