source "$HOME/auto-07p/cmds/auto.env.sh"
cd /Users/cris.hasan/Desktop/online_github_repositories/Indicator_system_modelling/paper_simulations/AUTO/non
auto

pgm = "non"
print "\nDemo %s is started"%pgm
print "\n%s : compute a solution branch"%pgm
r1 = run(e=pgm,c=pgm, ICP = [1,11])
save(r1,pgm)

print "\n***Clean the directory***"
cl()

