function shuffle(a,n,k) {
   a[1]=1; for(i=2;i<=k+1;i++){
     j=int(rand()*(n-i))+i
     if(j in a) a[i]=a[j]
     else a[i]=j
     a[j]=i;
   }
}

BEGIN {srand()}
FS="\t"
NR==1 {shuffle(ar,NF,ncols)}
  {for(i=1;i<=ncols+1;i++) printf "%s", $(ar[i]) FS; print ""}

