function MDK = mockMovieData(MD, idx)
% idx is the absolute idex already transformed from [col# row# site#],
% passed from value tni in movieViewer.viewsite
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
fileNamest = MD.channels_(1,1).hcsPlatestack_;
iFramex = zeros(length(idx), 4);
for iNiq = 1:max(size(idx))
    iT = 0;
    for iR = 1:size(fileNamest,1)
        for iC = 1:size(fileNamest,2)
            iW = size(fileNamest{iR, iC},2);
            iT = iT + iW;
            if idx(iNiq) <= iT && idx(iNiq) > iT - iW
                iFramex(iNiq,1) = iR; iFramex(iNiq,2) = iC;
                iFramex(iNiq,3) = idx(iNiq) - (iT - iW);
                iFramex(iNiq,4) = idx(iNiq);
            end
        end
    end
end
ch = Channel(MD.channels_(1,1).channelPath_, 'hcsPlatestack_', 1);
for i = 1:length(MD.channels_)
    hcsPlsmockcache = cell(1,size(iFramex, 1));
    for k = 1:size(iFramex,1)
    hcsPlsmockcache{k} = MD.channels_(1,i).hcsPlatestack_{iFramex(k,1),iFramex(k,2)}{iFramex(k,3)};
    end
    ch(1,i).hcsPlatestack_ = hcsPlsmockcache;
end
    mockMDname = MD.channels_(1,1).getGenericName(hcsPlsmockcache{1}, 'site_on');
    if max(size(hcsPlsmockcache)) ~= 1
        mockMDname = strcat('control begin with ', mockMDname);
    end
    mkdir(MD.outputDirectory_,'controls')
    MDK = MovieData(ch, [MD.outputDirectory_ filesep 'controls' filesep mockMDname]);
    MDK.movieDataPath_ = MD.movieDataPath_;
    %load([MD.movieDataPath_ filesep MD.movieDataFileName_]);
    MDK.mockMD_.parent = MD; %work as duplication
    MDK.mockMD_.index = iFramex;
end

    
