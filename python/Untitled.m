% Masukkan string
% list_gambar = {};
% for i = 1 : 6322
%     list_g = data{1};
%     ajj = string(list_g(i));
%     akk = strrep(ajj,'.','');
%     akk = strrep(akk,',','');
%     akk = char(akk);
%     akk = akk(2:end-1);
%     newStr = erase(akk, "jpg");
%     list_gambar{1,i} = newStr;
% end


% list_prior_que = [];
% abb = priorindex_queries;
% for i = 1 : 70 
%     
% 
%     all = imlist{abb(i)};
%     aha = find(strcmp(list_gambar, all));
%     list_prior_que = [list_prior_que ; aha];
% end


for i = 1 : 70
    
    aii = list_gambar_bbx{i};
    akk = int32(aii);
    gnd(i).bbx = akk;
end