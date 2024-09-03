function [starti, startsw] = getindexstart(filename)
% % generate a-z string
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
for n = 1:26
    strr{n} = char(n -1 + 'a');
end
% generate 0-9 string
for n = 1:10
    stnum{n} = num2str(n-1);
end
index_char = regexp(filename, strr, 'ignorecase');
startsw = []; starti = [];
for i = 1:length(index_char) %it does not check the 'tif'
    if ~isempty(index_char{i})
        while ~isempty(index_char{i})
            if index_char{i}(1)+3<length(filename)
                if sum(strcmp(stnum, filename(index_char{i}(1)+1))) == 0
                    index_char{i}(1) = [];
                elseif sum(strcmp(stnum, filename(index_char{i}(1)+2))) > 0 ...
                        && sum(strcmp(stnum, filename(index_char{i}(1)+3))) == 0 ...
                        && isempty(starti)...
                        && strcmp('_', filename(index_char{i}(1)-1))
                    starti = index_char{i}(1);
                elseif sum(strcmp({'s', 'w', 'S', 'W'}, filename(index_char{i}(1)))) > 0 ...
                        && sum(strcmp(stnum, filename(index_char{i}(1)+1))) > 0
                    startsw = [startsw index_char{i}(1)];
                    break;
                else
                    index_char{i}(1) = [];
                end
            else
                index_char{i}(1) = [];
            end
        end
    end
end
