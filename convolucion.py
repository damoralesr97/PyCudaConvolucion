import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import driver, compiler, gpuarray, tools

import numpy as np
import math

from PIL import Image

from flask import Flask, render_template, request, send_file
import os
from io import BytesIO




def obtenerKernel(sigma, MASK_SIZE):
    pico = MASK_SIZE//2
    factorNorm = 0.0
    kernel = np.empty((MASK_SIZE*MASK_SIZE),dtype=np.double)
    for i in range (0,MASK_SIZE):
        for j in range (0,MASK_SIZE):
            x = i - pico
            y = j - pico
            kernel[MASK_SIZE*i+j]= math.exp(-((x*x)+(y*y))/(2*sigma*sigma))
            factorNorm=kernel[MASK_SIZE*i+j]+factorNorm
    factorNorm=round(factorNorm,5)
    for i in range (MASK_SIZE):
        for j in range (MASK_SIZE):
            kernel[MASK_SIZE*i+j]= round(kernel[MASK_SIZE*i+j]/factorNorm,6)
    return kernel

    
    


IMAGE_FOLDER = os.path.join('static')

app = Flask(__name__)
app.config['UPLOAD FOLDER'] = IMAGE_FOLDER

@app.route("/", methods=['GET','POST'])
def practica():
    
    if request.method == 'POST':
        img = request.form['imagen']
        MASK_SIZE = request.form['mascara']
        sigma1 = request.form['sigma1']
        sigma2 = request.form['sigma2']
        
        cuda.init()
        device = cuda.Device(0)
        ctx = device.make_context()
        
        
        im     = Image.open(img).convert('RGB')
        kernel = obtenerKernel(float(sigma1),int(MASK_SIZE))
        kernel2 = obtenerKernel(float(sigma2),int(MASK_SIZE))
    
        W, H = im.size
        pixeles = np.empty((W*H),dtype=np.double)
        result  = np.empty((W*H),dtype=np.double)
        result2  = np.empty((W*H),dtype=np.double)
        result_gpu  = np.empty((W*H),dtype=np.double)
        result2_gpu  = np.empty((W*H),dtype=np.double)
    
        for i in range(W):
            for j in range(H):
                pixVal = im.getpixel((i, j))
                pixeles[W*j+i]=(0.3*pixVal[0])+(0.59*pixVal[1])+(0.11*pixVal[2])  
            
        pixeles_gpu= cuda.mem_alloc(pixeles.nbytes)
        kernel_gpu= cuda.mem_alloc(kernel.nbytes)
        kernel2_gpu= cuda.mem_alloc(kernel2.nbytes)
        result_gpu= cuda.mem_alloc(result_gpu.nbytes)
        result2_gpu= cuda.mem_alloc(result2_gpu.nbytes)


        cuda.memcpy_htod(pixeles_gpu, pixeles)
        cuda.memcpy_htod(kernel_gpu,kernel)
        cuda.memcpy_htod(kernel2_gpu,kernel2)
        cuda.memcpy_htod(result_gpu,result)
        cuda.memcpy_htod(result2_gpu,result2)

        mod = ("""
        __global__ void GPU_convolucion(double *matrix,double *filter, double *result) {
    
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            // Starting index for calculation
            int pico=%(M)s/2;
            int start_r = row - pico;
            int start_c = col - pico;
        
            // Iterate over all the rows
            for (int i = 0; i < %(M)s; i++) {
                // Go over each column
                for (int j = 0; j < %(M)s; j++) {
                    // Range check for rows
                    if ((start_r + i) >= 0 && (start_r + i) < (%(N)s*%(N)s)) {
                        // Range check for columns
                        if ((start_c + j) >= 0 && (start_c + j) < (%(N)s*%(N)s)) {
                            // Accumulate result
                            result[%(N)s*row+col] += matrix[(start_r + i) * %(N)s + (start_c + j)] * filter[i * %(M)s + j];
                        }
                    }
                }
            }
            if(result[%(N)s*row+col]>255)
                result[%(N)s*row+col]=255;
            if(result[%(N)s*row+col]<0)
                result[%(N)s*row+col]=0;
        }
        """)
        kernel_code = mod % {
            'M': MASK_SIZE,
            'N': W,
        }
        conv = compiler.SourceModule(kernel_code)
        func = conv.get_function("GPU_convolucion")

        func(pixeles_gpu,kernel_gpu,result_gpu,block=(32,32,1), grid=(32,32,1))
        func(pixeles_gpu,kernel2_gpu,result2_gpu,block=(32,32,1), grid=(32,32,1))

        cuda.memcpy_dtoh(result, result_gpu)
        cuda.memcpy_dtoh(result2, result2_gpu)

        resta=abs(result-result2)

        print ("-" * 80)
        print ("PIXELES (CPU):")
        print(pixeles[:10])
        print ("-" * 80)
        print ("KERNEL 1 (CPU):")
        print(kernel[:int(MASK_SIZE)])
        print ("-" * 80)
        print ("KERNEL 2 (CPU):")
        print(kernel2[:int(MASK_SIZE)])
        print ("-" * 80)
        print ("CONVOLUCION 1 (GPU):")
        print(result)
        print ("-" * 80)
        print ("CONVOLUCION 2 (GPU):")
        print(result2)
        print ("-" * 80)
        print ("DIFERENCIA DE GAUSSIANAS (CPU):")
        print(resta)

        matt = np.reshape(resta,(1024,1024))

        ig = Image.fromarray(np.uint8(matt))
        
        img_io=BytesIO()
        ig.save(img_io,"PNG")
        img_io.seek(0)
    
        ctx.pop()
        
        return send_file(img_io,mimetype='image/PNG');
    return render_template('index.html');

if __name__ == "__main__":
    app.run()




