#define ETH_ADDR_LEN 6

struct ethhdr{
    unsigned char dst_mac[ETH_ADDR_LEN];
    unsigned char src_mac[ETH_ADDR_LEN]; 
    unsigned short type; 
};

// 还有一个大小端的问题
// ntohs
// htons
struct iphdr{
    unsigned char version:4,
                    hrdlen:4;
    unsigned char tos;
    unsigned short totlen; //65535

    unsigned short id;
    unsigned short flag:3,
                   offset:13;
    unsigned char ttl;
    unsigned char proto;
    unsigned short check;

    unsigned int sip;
    unsigned int dip;

};


// UDP使用场景:
//      a. 大块的数据下载, UDP没有拥塞控制
//      b. 没有延迟确认机制, 发完之后不知道对方知不知道. 游戏实时性
struct udphdr{
    unsigned char sport;
    unsigned short dport;
    unsigned short length;
    unsigned short check;
};

struct udppkt{
    struct ethhdr eh;
    struct iphdr ip;
    struct udphdr udp;

    // 数据使用柔性数组(零长数组, sizeof出来是0)
    // 适用于长度已知, 知道肯定不会越界
    unsigned char patload[0];
};

// raw socket
// netmap --> 开源方案
// dpdk
// ebpf 
// pr_string

int main(){

}
