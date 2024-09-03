function [chplatestack, starti, startsw] = HCSplatestack(varagin)
chplatestack = [];
dirurl = varagin;
file_lists = dir(fullfile(dirurl, '*.TIF')); ct = 0;
lenf = zeros(1, 10);
for i2 = 1:10
lenf(i2) = length(file_lists(i2).name); % see if there is inconsistent double digit naming
end
% if lenf(10) ~= mean(lenf)
%     renameSingledigitfiles(dirurl, file_lists, max(lenf));
% end
%file_lists = dir(fullfile(dirurl, '*.TIF')); ct = 0;
%[starti, startsw] = getindexstart(file_lists(1).name);
%
% Copyright (C) 2024, Danuser Lab - UTSouthwestern 
%
% This file is part of uSegment3D_Package.
% 
% uSegment3D_Package is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% uSegment3D_Package is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with uSegment3D_Package.  If not, see <http://www.gnu.org/licenses/>.
% 
% 
h = waitbar(0, 'Loading HCS Images');
for i1 = 1:length(file_lists)
        waitbar(i1/length(file_lists));
        [starti, startsw] = getindexstart(file_lists(i1).name);
        wp = file_lists(i1,1).name(starti:max(startsw)+1); %well position
        wpv = double(wp(1))-64; %get numericle order from alphabetic sequence
        wph = str2double(wp(2:3));
        if min(abs(str2double(wp(min(startsw)-starti+3))-(0:9))) == 0 %see if site number is double digit.
        shp = str2double([wp(min(startsw)-starti+2),wp(min(startsw)-starti+3)]);
        chn = str2double(wp(max(startsw)-starti+2));
        else
            shp = str2double(wp(min(startsw)-starti+2));
            chn = str2double(wp(max(startsw)-starti+2));
        end
        ct = ct + 1;  
        if ct == 1
            wpvr = wpv; wphr = wph;
        end
        chplatestack{chn}{wpv-wpvr+1, wph-wphr+1}{shp} = [file_lists(i1,1).name];
end
close(h)

function renameSingledigitfiles(dirurl, file_lists, maxlength)
    [starti, startsw] = getindexstart(file_lists(1).name);
    ss = startsw(1);
    for i3 = 1:length(file_lists)
        if length(file_lists(i3).name) ~= maxlength
        movefile(strcat(dirurl, file_lists(i3).name), strcat(dirurl, file_lists(i3).name(1:ss), '0', file_lists(i3).name(ss+1:end)));
        end
    end
