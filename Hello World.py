#!/usr/bin/env python
# coding: utf-8

# In[1]:


print('hello world!')


# In[1]:


print ('hello world')


# In[2]:


print('hello world!')


# In[3]:


print ('hello world!')


# In[6]:


import urllib.request
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0101EN-SkillsNetwork/labs/Module%204/data/example1.txt'
filename='example1.txt'
urllib.request.urlretrieve(url, filename)


# In[7]:


get_ipython().system('wget -O /resources/data/Example1.txt https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0101EN-SkillsNetwork/labs/Module%204/data/example1.txt')


# In[9]:


example1='example1.txt'
file1=open(example1, 'r')


# In[10]:


file1.name


# In[11]:


file1.mode


# In[12]:


filecontent=file1.read()
filecontent


# In[13]:


print(filecontent)


# In[14]:


type(filecontent)


# In[15]:


file1.close()


# In[16]:


with open(example1, 'r') as file1:
    fileconent=file1.read()
    print(filecontent)


# In[17]:


file1.closed


# In[19]:


print(filecontent)


# In[21]:


with open(example1, 'r') as file1:
    print(file1.read(4))
    


# In[22]:


with open(example1, 'r')as file1:
    print(file1.read(4))
    print(file1.read(4))
    print(file1.read(7))


# In[23]:


with open(example1, 'r')as file1:
    print('firstline: ' + file1.readline())


# In[24]:


with open(example1, 'r')as file1:
    i=0;
    for line in file1:
        print('iteration',str(i), ': ', line)
        i=i+1


# In[1]:


import urllib.request
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0101EN-SkillsNetwork/labs/Module%204/data/example1.txt'
filename = 'Example1.txt'
urllib.request.urlretrieve(url, filename)


# In[2]:


get_ipython().system('wget -O /resources/data/Example1.txt https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0101EN-SkillsNetwork/labs/Module%204/data/example1.txt')


# In[5]:


example1='example1.txt'
file1=open(example1, 'r')


# In[6]:


file1.name


# In[7]:


file1.mode


# In[8]:


filecontent=file1.read
filecontent


# In[9]:


print(filecontent)


# In[10]:


type(filecontent)


# In[11]:


file1.close()


# In[ ]:




