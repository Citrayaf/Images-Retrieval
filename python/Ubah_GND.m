
list_gambar_junk = {};
for i = 1 : 70
    aka = {}; 

    all2 = qimlist{i};
    ahja = find(strcmp(list_gnd, all2));
        
    abb = gnd(i).junk;
    if length(abb) ~= 0
        for jj = 1 : length(abb)
            all = imlist{abb(jj)};
            aha = find(strcmp(list_gambar, all));
            aka{1,jj} = aha; 
        end
    end
    list_gambar_junk{ahja,1} = aka;
end

% list_gambar_bbx = {};
% for i = 1 : 70
% %     aka = {}; 
%     abb = gnd(i).bbx;
% 
%     all = qimlist{i};
%     aha = find(strcmp(list_gambar, all));
%         
%     list_gambar_bbx{aha,1} = abb;
% end