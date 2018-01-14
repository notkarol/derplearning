
/* note: by shifting and rotating as a part of
 cropping we can save time and detect oob faults */
/********Image shifting and rotating ********/

/********Patch selection and cropping *******/



/********Patch Resizing for NN input ********/


/* Gaussian Subsampler */
/* for a given i/o dimension a standard deviation span is set. 
    this span gives us a a general sample range.
    Then an RNG produces relative coordinates on that span.
    By adding the relative coordinates to the scalled absolute coordinates and doing OOB checks
    We get a source pixel to map to the destination. */

//TODO write unit tests. update install file to download and install google test framework

static int Image_Manip::sub_sampler( int src[], int out_dim)
{
   
    int x_src, y_src;
    float x_scale, y_scale; //scale factors from one image to another
    double x_shift, y_shift; //randomized selection 
    

    // Loop through every pixel in the output
    for( i = 0; i < out_dim[0]; i++){
        for( j = 0; j < out_dim[1]; j++){
            x_shift = Image_Manip::norm_resampler() //gaussian selector
            y_shift = Image_Manip::norm_resampler() 

            x_src = floor((i + x_shift) * x_scale);
            y_src = floor((j + y_shift) * y_scale);

            output[i,j] = src[x_src, y_src];
        }
    }

    return output;
}

/* Selects a random double bewteen 0-1 with mean .5
If the double falls outside of 0-1 range modulo division iteratively reduces it's size. */
static double Image_Manip::norm_resampler(){

    double rand = norm(generator);

    while(rand => 0.5) rand = rand % 10;
    while(rand < -0.5) rand = rand % 10;

    return rand + 0.5; //Shift rand from range [-0.5, 0.5) to [0, 1)
}