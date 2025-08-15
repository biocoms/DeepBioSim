library(HMP2Data)
m = momspi16S()
mtx = momspi16S_mtx
write.csv(mtx, "momspi16s.csv")

t2d = T2D16S()
t2d_mtx = T2D16S_mtx
write.csv(t2d_mtx, "t2d16s.csv")