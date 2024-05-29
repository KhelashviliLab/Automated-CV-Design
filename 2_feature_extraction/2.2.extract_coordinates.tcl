##### This Tcl code extracts the Cartesian coordinates of selected residues from a given trajectory. 
##### The coordinates will be used in the subsequent code for feature (e.g., pairwise distance) calculations. 

##### Author: MO (Initial Release Date: August 18, 2023) 
##### Updated by: MR (Latest Update: May 28, 2024) 

### STEP 1. Set output file names
# Edit according to your system (no changes needed if using our inputs):

set file_out "output_coords_sample_IFS_CA.csv"

# Open the output file for writing
set outfile [open $file_out w]

if {[catch {open $file_out w} outFile]} {
    puts "Error opening $file_out: $outFile"
    exit 1
}

### STEP 2. Load PSF and trajectory file 
# Edit according to your system (no changes needed if using our inputs):

mol load psf /lustre/orion/bip109/scratch/margaridarosa/mfsd2a_methods_simulations/OpenMM-Summit-Ensemble-master/mlcvs-final-GITHUB_May/1_input_MD_data/IFS.psf
mol addfile /lustre/orion/bip109/scratch/margaridarosa/mfsd2a_methods_simulations/OpenMM-Summit-Ensemble-master/mlcvs-final-GITHUB_May/1_input_MD_data/sample_IFS_trajectory.dcd first 0 last -1 step 1 waitfor -1

### Step 3. Extract coordinates
# Edit according to your system (no changes needed if using our inputs):

set nf [molinfo top get numframes]
set residList0 [[atomselect top "(protein and resid 39 to 68 75 to 101 110 to 129 137 to 167 171 to 202 233 to 263 292 to 318 328 to 353 358 to 374 382 to 416 424 to 450 465 to 491) and name CA"] get resid]

# Do not edit below this line (unless changing to res-res distances) 

set residList [lsort -unique -integer -increasing $residList0]
set minResid [tcl::mathfunc::min {*}$residList]
set maxResid [tcl::mathfunc::max {*}$residList]
set minIndex [lsearch $residList $minResid]
set maxIndex [lsearch $residList $maxResid]

puts "minIndex = $minIndex, maxIndex = $maxIndex, Length = [llength $residList]"

set xyzList {x y z}

# Write the header to the output file
for {set m $minIndex} {$m <= $maxIndex} {incr m} {
        set n [lindex $residList $m]
        foreach elem $xyzList {
                puts -nonewline $outfile "res$n.$elem,"
        }
        unset n
}
puts -nonewline $outfile "\n"

for {set i 1} {$i < $nf} {incr i} {
        set distList {}
        animate goto $i
        for {set p $minIndex} {$p <= $maxIndex} {incr p} {
                set pGly [atomselect top "protein and resid [lindex $residList $p] and name CA"]
                if {[$pGly get resname] == "GLY"} {
                        set sel1 [atomselect top "protein and resid [lindex $residList $p] and name CA"]
                        $sel1 frame $i
                        $sel1 update
                } else {
                        set sel1 [atomselect top "protein and resid [lindex $residList $p] and name CA"]
                        $sel1 frame $i
                        $sel1 update
                }
                set xCoord [lindex [measure center $sel1] 0]
                lappend distList $xCoord
                set yCoord [lindex [measure center $sel1] 1]
                lappend distList $yCoord
                set zCoord [lindex [measure center $sel1] 2]
                lappend distList $zCoord

                $sel1 delete
                $pGly delete
                unset xCoord
                unset yCoord
                unset zCoord
        }
        foreach elem $distList {
                puts -nonewline $outfile "$elem,"
        }
        unset -nocomplain distList
        puts -nonewline $outfile "\n"
}

unset nf
unset -nocomplain residList0
unset -nocomplain residList
unset -nocomplain xyzList
unset minResid
unset maxResid
unset minIndex
unset maxIndex

# Close the output file
close $outfile

