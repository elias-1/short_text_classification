function r = Datalocation(M,N,range1,range2)
i = 1;
r(1) = floor(1+(range1-1)*rand(1));
while r(i)+M-1 < N+1
    i = i + 1;
    t = floor(1 + (range2 - 1)*rand(1));
    r(i) = r(i-1) + M -1 + t;
end
r = r(1:i-1);