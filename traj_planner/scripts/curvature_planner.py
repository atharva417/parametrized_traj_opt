from math import factorial as f
import numpy as np
from scipy.linalg import block_diag
from qpsolvers import solve_qp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class GET_KAPPA:
    def __init__(self, x, y, n=8):
        self.n = n
        self.m = len(x)-1
        m = self.m
        #print(m)
        self.x = x
        self.y = y
        self.t = self.time_array()
        #self.t = [0.1,0.8,0.85,1.5]
        self.z = [1]*(m+1)
        self.q=np.zeros(shape=(n*m,1)).reshape((n*m,))
        self.G=np.zeros(shape=((4*m)+2,n*m))
        self.h=np.zeros(shape=((4*m)+2,1)).reshape(((4*m)+2,))
        b_x = np.array([x[0],0,0,x[m],0,0])
        b_x = np.append(b_x, x[1:m])
        b_x = np.append(b_x, np.zeros(shape=(3*(m-1))))
        b_y = np.array([y[0],0,0,y[m],0,0])
        b_y = np.append(b_y, y[1:m])
        b_y = np.append(b_y, np.zeros(shape=(3*(m-1))))
        b_z = np.array([self.z[0],0,0,self.z[m],0,0])
        b_z = np.append(b_z, self.z[1:m])
        b_z = np.append(b_z, np.zeros(shape=(3*(m-1))))
        self.b_x = b_x
        self.b_y = b_y
        self.b_z = b_z
        self.v = 0.5
        
        self.form_Q()
        self.form_A()

    def time_array(self):
        t = [0.1]
        for i in range(1,self.m+1):
            dist = np.sqrt((self.x[i]-self.x[i-1])**2 + (self.y[i]-self.y[i-1])**2 + (self.z[i]-self.z[i-1])**2)
            ti = dist/self.v
            t.append(t[-1]+ti)
        return t

    def form_Q(self):
        Q_list = []
        for l in range(1,self.m+1):
            
            Q_i=np.zeros(shape=(self.n,self.n))
            for i in range(self.n):       
                for j in range(self.n):
                    if (((i>3) and (j<=3))) or (((j>3) and (i<=3))):
                        Q_i[i][j]=0
                    elif (i<=3) and (j<=3):
                        Q_i[i][j]=0
                    else:
                        r,c=i+1,j+1
                        Q_i[i][j]= (f(r)*f(c)*(pow(self.t[l],r+c-7)-pow(self.t[l-1],r+c-7)))/(f(r-4)*f(c-4)*(r+c-7))
            Q_list.append(Q_i)
        Q = Q_list[0]
        Q_list.pop(0)
        for i in Q_list:
            Q=block_diag(Q,i)
        self.Q=Q+(0.0001*np.identity(self.n*self.m))
        print(type(self.Q),'typeofQ')

    def form_A(self):
        n = self.n
        m = self.m
        t = self.t
        A=np.zeros(shape=((4*m)+2,n*m))

        for j in range(n*m):
                if j>=n:
                    A[0][j],A[1][j],A[2][j]=0,0,0
                else:
                    A[0][j],A[1][j],A[2][j]=pow(t[0],j),j*pow(t[0],j-1),j*(j-1)*pow(t[0],j-2)

        for j in range(n*m):
                if j<n*(m-1):
                    A[3][j],A[4][j],A[5][j]=0,0,0
                else:
                    h=n*(m-1)
                    A[3][j],A[4][j],A[5][j]=pow(t[m],j-h),(j-h)*pow(t[m],j-h-1),(j-h)*(j-h-1)*pow(t[m],j-h-2)
        z=[]
        for i in range(1,m):
            h=[]
            for j in range(n*m):
                if (j<((i-1)*n)) or (j>=(i*n)):
                    h.append(0)
                else:
                    h.append(pow(t[i],j-((i-1)*n)))
            z.append(h)    
        A[6:(6+m-1)]=z
        pva_const=[]
        for i in range(1,m):
            x_i,v_i,a_i=[],[],[]
            for j in range(n*m):
                if (j<((i-1)*n)) or (j>=((i+1)*n)):
                    x_i.append(0)
                    v_i.append(0)
                    a_i.append(0)
                    
                elif (j<((i)*n)) and (j>=((i-1)*n)):
                    x_i.append(pow(t[i],j-((i-1)*n)))
                    v_i.append((j-((i-1)*n))*pow(t[i],j-1-((i-1)*n)))
                    a_i.append((j-1-((i-1)*n))*(j-((i-1)*n))*pow(t[i],j-2-((i-1)*n)))
                else:
                    x_i.append((-1)*pow(t[i],j-((i)*n)))
                    v_i.append((-1)*(j-((i)*n))*pow(t[i],j-1-((i)*n)))
                    a_i.append((-1)*(j-1-((i)*n))*(j-((i)*n))*pow(t[i],j-2-((i)*n)))
            pva_i=[x_i,v_i,a_i]        
            pva_const=pva_const+pva_i
        A[(6+m-1):]=pva_const
        self.A = A
        self.solve()

    def solve(self):
        self.p_x=solve_qp(self.Q, self.q,self.G,self.h, self.A, self.b_x)
        self.p_y=solve_qp(self.Q, self.q,self.G,self.h, self.A, self.b_y)
        self.p_z=solve_qp(self.Q, self.q,self.G,self.h, self.A, self.b_z)

    def plot(self):
        plt.figure(figsize=(10,5))
        ax = plt.axes(projection ='3d')

        ax.scatter(self.x, self.y, self.z, 'b',marker='o')
        for v in range(self.m):
            w,u,a=[],[],[]
            
            r=np.linspace(self.t[v],self.t[v+1],100)
            for i in range(100):
                g,e,f=0,0,0
                for j in range(self.n*v,(v+1)*self.n):
                    g=g+(self.p_x[j]*pow(r[i],j-(self.n*v)))
                    e=e+(self.p_y[j]*pow(r[i],j-(self.n*v)))
                    f=f+(self.p_z[j]*pow(r[i],j-(self.n*v)))
                w.append(g)
                u.append(e)
                a.append(f)
            ax.plot3D(w, u, a, 'r')
        plt.show()

    def get_trajectory_var(self):
        for v in range(self.m):
            w,u,a=[],[],[]
            w_v,u_v,a_v=[],[],[]
            w_a,u_a,a_a=[],[],[]
            
            r=np.arange(self.t[v],self.t[v+1],self.dt)
            for i in range(0,r.shape[0]):
                g,g_v,g_a,e,e_v,e_a,f,f_v,f_a=0,0,0,0,0,0,0,0,0
                for j in range(self.n*v,(v+1)*self.n):
                    g=g+(self.p_x[j]*pow(r[i],j-(self.n*v)))
                    e=e+(self.p_y[j]*pow(r[i],j-(self.n*v)))
                    f=f+(self.p_z[j]*pow(r[i],j-(self.n*v)))

                    g_v=g_v+((j-(self.n*v))*self.p_x[j]*pow(r[i],j-1-(self.n*v)))
                    e_v=e_v+((j-(self.n*v))*self.p_y[j]*pow(r[i],j-1-(self.n*v)))
                    f_v=f_v+((j-(self.n*v))*self.p_z[j]*pow(r[i],j-1-(self.n*v)))

                    g_a=g_a+((j-(self.n*v))*(j-1-(self.n*v))*self.p_x[j]*pow(r[i],j-2-(self.n*v)))
                    e_a=e_a+((j-(self.n*v))*(j-1-(self.n*v))*self.p_y[j]*pow(r[i],j-2-(self.n*v)))
                    f_a=f_a+((j-(self.n*v))*(j-1-(self.n*v))*self.p_z[j]*pow(r[i],j-2-(self.n*v)))


                w.append(g)
                w_v.append(g_v)
                w_a.append(g_a)

                u.append(e)
                u_v.append(e_v)
                u_a.append(e_a)

                a.append(f)
                a_v.append(f_v)
                a_a.append(f_a)
  
            self.x_path.extend(w)
            self.x_dot_path.extend(w_v)
            self.x_dot_dot_path.extend(w_a)

            self.y_path.extend(u)
            self.y_dot_path.extend(u_v)
            self.y_dot_dot_path.extend(u_a)

            self.z_path.extend(a)
            self.z_dot_path.extend(a_v)
            self.z_dot_dot_path.extend(a_a)

            self.psi_path=np.arctan2(self.y_dot_path,self.x_dot_path)
        self.x_dot_path = np.array(self.x_dot_path)
        self.x_dot_dot_path = np.array(self.x_dot_dot_path)
        self.y_dot_path = np.array(self.y_dot_path)
        self.y_dot_dot_path = np.array(self.y_dot_dot_path)
        self.kappa = (self.y_dot_dot_path * self.x_dot_path - self.x_dot_dot_path * self.y_dot_path) / (np.power(self.x_dot_path ** 2 + self.y_dot_path ** 2,1.5))
        #print(self.z_path)

        # return self.x_path,self.x_dot_path,self.x_dot_dot_path,self.y_path,self.y_dot_path,self.y_dot_dot_path,self.z_path,self.z_dot_path,self.z_dot_dot_path,self.psi_path
