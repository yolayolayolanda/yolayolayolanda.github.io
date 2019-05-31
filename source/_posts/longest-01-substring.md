---
title: Longest 01 Substring
date: 2019-05-31 13:35:03
categories: algorithms
mathjax: true
toc: true
---

#### Problem

##### Input 

A 01 string

##### Output

The longest 01 substring where number of 1 = number of 0.

#### Algorithm

In a substring, number of 1 = number of 0 $\iff $ `count[1] - count[0]` has no change at the beginning and the end of this substring.

<!-- more -->

So we can use a list `B` to record the difference, and a map to record the position of `B[i]`, so that we can calculate the length of the substring  according to where do we get the same difference last time.

#### Time Complexity
$O(n)$

#### Implementation language 

Python

#### Data Structure

```python
count = [0, 0]			# counter of number of 0s and 1s
B = [0] * len(s)		# recorder of differences
dic = {}  					# Map from differences to positions						
lengest = 0					# current maximum length
```

#### Code

```python
def lengest01SubStr(self, s):
    count = [0, 0]
    B = [0] * len(s)
    dic = {}  
    lengest = 0
    
    for i in range(len(s)):
        count[int(s[i])] += 1
        B[i] = count[0] - count[1] 
        # difference is 0 means the substring is from the beginning of the string
        if B[i] == 0:  
            lengest = i + 1
            continue
        # when the difference can be found in the dic, it means there is a qualified substring finished
        if B[i] in dic:
            lengest = max(lengest, i - dic[B[i]]) 
        # else append the new difference
        else:
            dic[B[i]] = i
    return lengest
```

#### Test

```python
a = '1011010'
b = '10110100'
c = '00100110010011'
print(lengest01SubStr(a))
print(lengest01SubStr(b))
print(lengest01SubStr(c))


---------------------------------------------
6
8
12
```
