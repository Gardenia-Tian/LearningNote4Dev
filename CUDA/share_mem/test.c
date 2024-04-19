#include<stdio.h>

int main(){
    int a[32] = {0};
    for(int i = 0;i<32;i++){
        printf("%d ", i %2 == 0? i : i - 1);
    }
    printf("\n");
}