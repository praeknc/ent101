# Primary Market Research: Where Data Center Power Engineers Hang Out Online, and How to Reach Them

*Prepared 2026-10-05. Sources are cited inline and listed at the end.*

---

## 1. Executive summary

**Research question.** Where do the people who build and validate grid connections for AI data centers gather online? The two target groups are:

1. Power and electrical infrastructure engineers at neocloud operators: CoreWeave, Crusoe, Nebius, Lambda.
2. Grid-connection and critical-facilities engineers at developers and colocation operators: Lancium, Equinix, Digital Realty, Virtus, Kao Data.

The goal is to start peer-to-peer research conversations with them about EMT modeling and getting connected to the grid faster.

**Search parameters.** We searched for these terms: PSCAD, EMT, data center load model, ride-through, NOGRR 282, NPRR 1308, NERC Level 3 alert, Large Loads Task Force, LLWG, CIFP, flexible connection, and NESO connections reform. We crossed them with these platforms: Reddit, Discord, Slack, LinkedIn, groups.io, vendor forums, IEEE/CIGRE/ESIG working groups, and industry associations. Geographic focus is ERCOT and PJM in the US and NESO in Great Britain.

**Why this is a good time to reach out (fall 2026).** Several deadlines have just passed, so the topic is on these engineers' minds.

- **ERCOT NPRR 1308 / NOGRR 282.** NPRR 1308 creates the "Large Electronic Load" category: a site above 75 MW where at least 50% of the load is computational. It was approved on 2026-07-09 and took effect on 2026-08-01. NOGRR 282 sets the voltage and frequency ride-through rules for these loads. Industry commentators estimate compliance costs of about **$0.5M–$1M per MW**. ERCOT counts **28 events since January 2023** in which computational load dropped by at least 100 MW during a grid disturbance.
- **NERC Level 3 "Essential Actions" alert** (issued 2026-05-04). This is NERC's highest alert level. It followed events in which more than 1,000 MW of computational load tripped within seconds. Registered entities had to respond by **2026-08-03**. Utilities are now passing those requirements on to their data center customers.
- **Validated EMT models are now an interconnection requirement.** In April 2026, ERCOT published an open-source PSCAD model of a double-conversion UPS AI data center. ESIG published *Large Load Modeling for Dynamic Studies* the same month. SPP has its own EMT model requirements. Engineering firms are hiring "PSCAD – Data Centers" engineers.
- **Queues remain long.**
  - PJM's CIFP board decision (2026-01-16) created an Expedited Interconnection Track for projects that bring their own new generation.
  - In Great Britain, NESO counts about 140 data centers (around 50 GW) in its queue. Ofgem is considering making flexible or non-firm connections a condition of connection.

**Main finding.** The target engineers rarely post in public forums. They concentrate in three kinds of places:

1. **Stakeholder working groups** with public rosters and minutes (ERCOT LLWG, NERC LLWG, ESIG LLTF, IEEE PES, CIGRE, OCP).
2. **Private LinkedIn groups and Sales Navigator searches.**
3. **A few practitioner forums** (the PSCAD Forum, Eng-Tips, and r/PowerEngineering).

Discord and Slack are useful for reaching people one step removed from the target, such as GPU-infrastructure engineers and DER/grid generalists. They are a poor fit for direct outreach to the core persona. **Recommendation:** put about 60% of effort into working-group rosters and LinkedIn, 25% into technical forums (PSCAD, Eng-Tips, Reddit), and 15% into Slack and Discord.

> **Data caveat.** Member counts marked ✅ come from a cited third-party source. Counts marked ~ are estimates. Reddit, ERCOT and some community sites block automated fetching from this research environment, so re-check live counts before using them.

---

## 2. Watering holes by category

### 2A. Stakeholder and standards working groups (best fit)

Members of these groups are working on exactly the compliance problem we care about. Meeting minutes and attendee lists name companies and often individuals, which makes them the best source of prospect lists.

| # | Community / source | Platform | Size / activity | URL / access | Fit and how to approach |
|---|---|---|---|---|---|
| 1 | **ERCOT Large Load Working Group (LLWG)** | ISO stakeholder WG (public meetings plus email list) | Meets roughly monthly. Recent agendas cover Batch Zero, large computational load reliability and ride-through. Speakers have included Google, Oncor, Skybox and others ✅ | https://www.ercot.com/committees/tac/llwg | **Very high.** Ground zero for NOGRR 282. Read the meeting materials and note which companies present or comment, then contact those engineers and cite the specific slide or comment. Asking a question in the meeting is acceptable; pitching is not. |
| 2 | **NERC Large Loads Working Group** (formerly LLTF, under RSTC) | Standards body WG | Coordinates with the Data Center Coalition and ERCOT LLWG. Drove the Level 3 alert ✅ | https://www.nerc.com (RSTC → LLWG). See also the LLTF white paper | **High.** Utility planners and data center registrants attend. Use the attendee rosters in RSTC agenda packages for prospecting. |
| 3 | **ESIG Large Loads Task Force** (Modeling team led by EPRI's Parag Mitra) | Industry task force plus workshops | 7–8 project teams. Fall 2026 technical workshop ✅ | https://www.esig.energy/working-groups/large-loads/ | **Very high** for EMT modeling. Attend the workshops; the "Large Load Modeling, Testing & Interconnection Requirements" workshop matches our topic exactly. Contact speakers afterwards and reference their slides. |
| 4 | **EPRI DCFlex** | Utility/hyperscaler collaborative | 49 participants, including Google, Meta, NVIDIA, Compass, QTS, ERCOT, PJM and Schneider ✅ | https://www.epri.com (search "DCFlex") | **High** for the secondary (flexible connection) persona. Membership is gated, but published demonstration results name participants to contact. |
| 5 | **IEEE PES Substations Committee – WG T1 (Data Centers)** | IEEE PES WG | Writing a guide for large-load substations up to 800 kV / 15 GW ✅ | https://cmte.ieee.org/pes-substations/sct0/wgt1/ | **Medium–high.** Open to volunteers. Joining gives credible peer access to utility and owner engineers. |
| 6 | **IEEE PES Task Force on Data Center Growth & Grid Readiness**, plus local **IEEE PES chapters** (Dallas, Houston, Austin, N. Virginia) | IEEE PES TF and chapters | Technical report published May 2025. Chapters hold monthly vTools events ✅ | https://events.vtools.ieee.org (filter by PES and your region) | **Medium–high.** Offer to present at a local chapter on "EMT models for large electronic loads"; this produces warm inbound leads. |
| 7 | **CIGRE WG C4.81** – *Best practice modelling & study methods for large inverter-based loads* (approved 2026) | CIGRE study committee WG | New WG with an open call for experts ✅ | https://www.cigre.org/article/home/cigre-active-working-groups--call-for-experts | **High** for EMT specialists worldwide, including the UK/EU colocation persona. Join through your national committee (CIGRE-USNC or CIGRE UK). |
| 8 | **OCP Data Center Facility – Power Distribution sub-project** | OCP WG plus groups.io mailing list | Public and open to all. Led by Paolo Catapane and Joshua Buzzell ✅ | https://www.opencompute.org/community/power-distribution · https://ocp-all.groups.io/ | **Medium.** Focuses on what happens inside the fence (LVDC, rack power), but neocloud power leads attend. Join the list and the monthly calls. |
| 9 | **PJM CIFP – Large Load Additions** (and its follow-on stakeholder groups) | ISO stakeholder process | Board decision issued 2026-01-16 ✅ | https://www.pjm.com (Committees & Groups → CIFP-LLA) | **High** for the secondary persona (BYONG, expedited track, curtailment). The posted stakeholder comment letters list developer contacts. |

### 2B. Independent engineering forums and vendor communities

| # | Community / source | Platform | Size / activity | URL / access | Fit and how to approach |
|---|---|---|---|---|---|
| 10 | **PSCAD Community Forum** (Manitoba Hydro International) | Vendor forum | PSCAD has 30,000+ users worldwide ✅. The forum supports file and example sharing | https://www.pscad.com (Support → Forum) | **Very high** for model builders. Post technical questions, for example about convergence and runtime with the ERCOT UPS data center model. Once a thread is useful, DM the people who replied. |
| 11 | **ERCOT / Zenodo open-source AI data center PSCAD model** | Open model repository | Released 2026-04-24 ✅ | https://zenodo.org/records/21004140 | **High.** People who use this model are prospects. Watch citations, forks and follow-on papers on arXiv and IEEE DataPort to find them. |
| 12 | **IEEE DataPort – EMT Modeling of Data Centers** (PSU → rack → 200 MW building library) | Dataset community | Open access ✅ | https://ieee-dataport.org/open-access/electromagnetic-transient-emt-modeling-data-centers | **Medium–high.** Contact the authors and the people who download or cite it. |
| 13 | **Eng-Tips – "Electric power & transmission & distribution" forum** | Independent forum (since 1997) | The largest professional engineering forum ✅. The power forum is active daily | https://www.eng-tips.com/ | **Medium–high.** Senior utility and EPC engineers post here. Ask genuine technical questions; the forum culture is hostile to self-promotion. |
| 14 | **emt-tools.com / ResearchGate "PSCAD" topic** | Independent Q&A | ResearchGate has 120+ PSCAD Q&A threads ✅ | https://emt-tools.com/ · https://www.researchgate.net/topic/PSCAD | **Medium.** Mostly academics and consultants, who can be useful referral nodes. |
| 15 | **Open Energy Modelling Initiative forum** | Discourse forum | Active, research-oriented ✅ | https://forum.openmod.org/ | **Low–medium.** Planning modelers rather than EMT specialists. Use it for capacity-expansion context only. |
| 16 | **ServeTheHome Forums** | Independent forum | Active. Covers AI server power delivery and liquid cooling ✅ | https://forums.servethehome.com/ | **Medium** for neocloud cluster infrastructure engineers, who deal with rack and PSU behavior (the source of load oscillations). |
| 17 | **7x24 Exchange** (27 chapters, 375+ member companies) | Association plus chapters | ✅ | https://www.7x24exchange.org · LinkedIn: 7x24 Exchange International | **High** for critical-facilities and developer engineers. Chapter meetings (DC, Lone Star, Metro NY and others) are the best in-person venue. |
| 18 | **AFCOM / Data Center World** | Association plus chapters | Founded 1980. Local chapters ✅ | https://afcom.com | **Medium.** Skews toward operations and facilities. Useful for the colocation persona. |

### 2C. Reddit

| # | Community | Size | URL | Fit and how to approach |
|---|---|---|---|---|
| 19 | **r/PowerEngineering** | ~50–100k (unverified, check the sidebar) | https://www.reddit.com/r/PowerEngineering/ | **High.** Utility planners, protection engineers and EMT consultants post here. Threads about "PSCAD for data centers" and "large load interconnection" do well. Post a discussion question such as "How is your utility handling NOGRR 282 EMT model reviews?" and DM thoughtful commenters a day later. |
| 20 | **r/datacenter** | **43.7k** ✅ | https://www.reddit.com/r/datacenter/ | **Medium.** Many technicians, but critical-facilities engineers and electrical leads are active. Frame posts around UPS ride-through, generator transfer and AI load swings. |
| 21 | **r/ElectricalEngineering** | **~370k** ✅ | https://www.reddit.com/r/ElectricalEngineering/ | **Low–medium.** Broad audience. Use it only for widely interesting posts, for example how GPU training jobs create grid oscillations. |
| 22 | **r/energy, r/electricians (industrial), r/sysadmin (HPC threads)** | Large | — | **Low.** Monitor with keyword alerts (F5Bot or Syften) for "ERCOT", "PSCAD", "ride-through" and "CoreWeave power". |

### 2D. Discord

| # | Community | Size | URL | Fit and how to approach |
|---|---|---|---|---|
| 23 | **GPU MODE** | **~26.8k** ✅ | https://discord.com/invite/g5gAD2vFcm | **Low–medium (adjacent).** GPU and kernel engineers at Lambda, CoreWeave, Nebius and Crusoe. Good for questions about how training synchronization drives power swings and for referrals to their own power teams. |
| 24 | **PyPSA Discord** | Active, with weekly developer calls ✅ | https://pypsa.org (Discord link) | **Low–medium.** Grid planners and modelers. Useful for flexibility and capacity questions. |
| 25 | **"Data Center Dude" infrastructure community** | 600+ ✅ | https://datacenterdude.com/community/ | **Medium.** Small, vendor-neutral practitioner community. A good place to try out interview questions. |

### 2E. Slack

| # | Community | Size | URL | Fit and how to approach |
|---|---|---|---|---|
| 26 | **Work on Climate** | 10,000+ ✅ | https://workonclimate.org/ | **Medium.** Has #grid, #energy-storage and city channels. Many grid-software and utility people. Post a well-scoped research request. |
| 27 | **DER Task Force** | ~2,000 ✅ | https://www.dertaskforce.com/ | **Medium–high** for the flexible connection persona. Grid-edge and utility engineers who care about curtailment and flexible interconnection. |
| 28 | **MCJ Collective** | ~2,000 (paid, application required) ✅ | https://mcj.vc/members | **Medium.** Founders and investors with good introductions to developer teams. Ask for warm intros here. |
| 29 | **ClimateAction.tech** | 10,000+ ✅ | https://climateaction.tech/ | **Low–medium.** Software engineers who work on data center energy. Use it for referrals. |

### 2F. LinkedIn groups and pages

| # | Group / page | Size | Access | Fit |
|---|---|---|---|---|
| 30 | **Data Center Engineering** (private, since 2008) | 18,000+ ✅ | LinkedIn → Groups → search "Data Center Engineering" | **High.** Discussion of data center power infrastructure. |
| 31 | **DatacenterDynamics – Global Discussion** | 16,000+ ✅ | Search "DatacenterDynamics Global Discussion" | **Medium–high.** |
| 32 | **Data Center Advisors** | ~3,000 ✅ | Search "Data Center Advisors" | **Medium.** |
| 33 | **IEEE Power & Energy Society** (official group and page) | Large | Search "IEEE Power & Energy Society" | **Medium–high.** Filter group members by title. |
| 34 | **7x24 Exchange International** and chapter pages (e.g. DC Chapter) | — | linkedin.com/company/7x24-exchange-international | **High.** Event attendees are prospects. |
| 35 | **PSCAD** company page followers / **Manitoba Hydro International** | — | linkedin.com/company/pscad | **High.** "Followers" plus title filters give an EMT-literate audience. |

> **Tip.** LinkedIn group membership unlocks direct messages to fellow group members without spending InMail credits. Join groups 30, 31, 33 and 34 before you start sending messages.

---

## 3. LinkedIn search strings and boolean filters

### 3A. Ready-to-use people search URLs (standard LinkedIn)

You can paste these directly. After the page loads, use **Current company** and **Locations** in the UI to narrow the results.

```
# Neocloud power engineers
https://www.linkedin.com/search/results/people/?keywords=%22power%20engineer%22%20AND%20(CoreWeave%20OR%20Crusoe%20OR%20Nebius%20OR%20%22Lambda%22)

# PSCAD / EMT + data center
https://www.linkedin.com/search/results/people/?keywords=(PSCAD%20OR%20EMT%20OR%20EMTP)%20AND%20(%22data%20center%22%20OR%20%22large%20load%22)

# ERCOT large-load interconnection people
https://www.linkedin.com/search/results/people/?keywords=(ERCOT%20OR%20NOGRR%20OR%20%22large%20load%22)%20AND%20(interconnection%20OR%20%22ride-through%22)

# Grid connection engineers at colo/developers
https://www.linkedin.com/search/results/people/?keywords=(%22grid%20connection%22%20OR%20%22power%20design%20manager%22%20OR%20%22utility%20interconnection%22)%20AND%20(Equinix%20OR%20%22Digital%20Realty%22%20OR%20Lancium%20OR%20Virtus%20OR%20%22Kao%20Data%22)
```

### 3B. Boolean strings (Sales Navigator / Recruiter keyword or title fields)

**Primary persona: neocloud power engineers**
```
Title: ("power engineer" OR "electrical engineer" OR "electrical infrastructure" OR "power systems" OR "data center power" OR "cluster infrastructure" OR "critical power" OR "energy infrastructure") NOT (sales OR recruiter OR intern OR "account executive")
Company: CoreWeave OR Crusoe OR Nebius OR "Lambda" OR Together OR "Vultr" OR Nscale OR "Fluidstack" OR "TensorWave" OR "Voltage Park" OR "IREN" OR "Applied Digital" OR "Core Scientific"
Keywords (optional): PSCAD OR EMT OR "PSS/E" OR PowerFactory OR "ride-through" OR ERCOT OR interconnection OR substation
```

**EMT specialists anywhere who touch data centers (consultants are good referral sources)**
```
Keywords: (PSCAD OR "PSCAD/EMTDC" OR EMTP OR "EMT model" OR "electromagnetic transient") AND ("data center" OR "large load" OR "large electronic load" OR "computational load" OR hyperscale)
Title: ("power systems engineer" OR "transmission planning" OR "interconnection engineer" OR "dynamic studies" OR "model validation")
```

**Secondary persona: developers and colocation**
```
Title: ("grid connection" OR "utility interconnection" OR "power design" OR "critical facilities engineer" OR "site energy" OR "energy procurement" OR "power strategy" OR "head of power" OR "VP power" OR "director of energy")
Company: Lancium OR Equinix OR "Digital Realty" OR Virtus OR "Kao Data" OR "QTS" OR "Compass Datacenters" OR "Vantage Data Centers" OR "Aligned" OR "Stack Infrastructure" OR "CyrusOne" OR "EdgeCore" OR "Prime Data Centers" OR "Skybox Datacenters" OR "Tract"
```

**Regulatory and market experts (ERCOT/PJM/NESO)**
```
Keywords: ("NOGRR 282" OR "NPRR 1308" OR "Large Electronic Load" OR LLWG OR "Batch Zero" OR CIFP OR BYONG OR "NERC Level 3" OR "large loads working group" OR "TM04+" OR "flexible connection" OR "non-firm connection")
```

**Users of competing or adjacent tools (signals interest in this problem)**
```
Keywords: ("Emerald AI" OR "Edgeton" OR "Ting" OR "DCFlex" OR "PowerFactory" OR "Simulink Simscape Electrical" OR "EMTP-RV") AND ("data center" OR "large load")
```

### 3C. Sales Navigator filters that improve results

- **Posted on LinkedIn in last 30 days**: about 3× higher reply rate.
- **Changed jobs in last 90 days**: new hires are actively building their tool stacks.
- **Groups** set to the groups in §2F: you can message these people for free.
- **School**: your alma mater, to support Template 4.
- **Seniority**: Senior, Manager or Director, so you reach practitioners who own decisions.
- **Saved-search alert** on "NOGRR 282 OR PSCAD AND data center" to catch new posts.

### 3D. Google X-ray fallbacks

```
site:linkedin.com/in ("PSCAD" OR "EMT") ("CoreWeave" OR "Crusoe" OR "Nebius" OR "Lambda")
site:linkedin.com/in ("grid connection" OR "interconnection") ("Equinix" OR "Digital Realty" OR "Kao Data" OR "Virtus")
site:linkedin.com/posts ("NOGRR 282" OR "NOGRR282") data center
site:ercot.com LLWG "comments" filetype:pdf      # stakeholder comment letters name engineers
```

---

## 4. Outreach templates

**Rules for every template**

- Present yourself as a researcher, not a vendor.
- Name one concrete technical fact so the reader knows you understand their work.
- Ask for 15 minutes.
- Offer an NDA.
- Never attach a pitch deck.
- Personalize the first line: the `[Hook]` field must refer to something they posted, presented or published.

### Template 1: Neocloud power engineer (LinkedIn / email)

**Subject options**
- `NOGRR 282 + PSCAD data-center models: 15 min research chat?`
- `How long is your EMT model review taking? (research, not sales)`

**LinkedIn connection note (≤300 characters):**
> Hi [First Name], I'm researching how neocloud power teams are handling EMT model validation for NOGRR 282 / the NERC Level 3 alert. I saw [Hook: your LLWG comment / post on UPS ride-through]. Would value 15 min of your perspective. Happy to share findings back.

**Email / follow-up message:**
> Hi [First Name],
>
> I'm [Your Name], a [your role, e.g. power systems researcher at MIT]. I'm running primary research with power engineers at GPU cloud operators on one specific bottleneck: **building and validating dynamic EMT load models fast enough to get to energization.**
>
> I'm reaching out because [Hook: your work on [Site/Project], your comment on [Topic], your talk at [Event]].
>
> What we keep hearing:
> - Utilities and ERCOT now expect **PSCAD models that represent UPS/PSU and rack-level behavior**. ERCOT's April 2026 reference model shows the level of detail they want. Getting from OEM black boxes to a model that passes review is taking [weeks/months].
> - **NOGRR 282 ride-through** is now in force, with estimates of $0.5–1M/MW to comply. The **NERC Level 3 alert** is pushing TOs to ask for more data on synchronized GPU load swings.
>
> Is that consistent with what you're seeing on [Company]'s [Region/Site] builds? I'd value **15 minutes** on:
> 1. Where model validation actually stalls (OEM data, utility review cycles, or tooling).
> 2. How you're handling ride-through at the UPS or BESS layer versus with grid-side mitigation.
>
> This is research, not a sales call. I'll share the anonymized findings across operators. If equipment parameters come up, **I'm happy to sign your NDA first.**
>
> Would [Day] or [Day] work? Or reply with a better time.
>
> Thanks,
> [Name] · [Title] · [LinkedIn / site]

**Follow-up after 4–5 days:**
> Hi [First Name], just floating this back up. Even a 2-line reply would help: *what's the single slowest step between "EMT model requested" and "model accepted"?* I'll send you the anonymized summary either way.

### Template 2: Data center developer or colocation grid-connection engineer (LinkedIn / email)

**Subject options**
- `Trading peak-hour flexibility for an earlier energization date? (research)`
- `[Campus Name] grid timeline: 15 min research question`

> Hi [First Name],
>
> I'm [Your Name], researching how data center developers are **getting energized faster** while connection queues are measured in years. I noticed [Hook: [Company]'s [MW] campus in [Location] / your comments on [PJM CIFP / NESO TM04+ / ERCOT Batch Zero]].
>
> We're comparing three approaches developers are using:
> - **Flexible / non-firm connections.** You accept curtailment in peak hours in exchange for an earlier date. Ofgem is now considering making this a condition of connection, and PJM's Expedited Interconnection Track rewards bringing your own generation.
> - **Bridging with on-site generation or BESS** until firm capacity arrives.
> - **Proving model quality early.** That means submitting validated EMT and ride-through models with the application so that study cycles aren't repeated.
>
> I'd love **15 minutes** to hear:
> 1. What curtailment terms (hours per year, notice time, MW) you'd accept for each year an energization date moves earlier.
> 2. Where your current timeline actually stalls: studies, upgrades, equipment, or permits.
>
> This is a research conversation with no pitch. I'll share the anonymized benchmark across developers. **NDA available** if you want to talk site-specific numbers.
>
> Would [Day/Time] work?
>
> Best,
> [Name] · [Title] · [Contact]

### Template 3: Community DM or post (Discord / Slack / Reddit)

**Public post (r/PowerEngineering, Work on Climate #grid, DER Task Force):**
> **How are you validating EMT models for large data center loads?**
> Asking for a research project, not selling anything. For anyone doing interconnection studies for large electronic loads (ERCOT NOGRR 282 or elsewhere):
> - Are you starting from ERCOT's open-source UPS data center PSCAD model, OEM black-box models, or building your own?
> - What's the slowest part: getting PSU/UPS parameters, utility review cycles, or simulation runtime at 200 MW+ scale?
> - Has anyone found a workable path around the queue, such as flexible or curtailable connection or bridging with on-site generation?
>
> I'll compile the answers and post a summary back here.

**DM to a helpful commenter (1–2 days later):**
> Hey [handle]! Your point about [specific detail, e.g. "PSU control loops interacting with weak-grid SCR"] was the most useful answer in the thread. I'm doing a small set of 15-min chats with engineers working on this. Would you be up for one sometime this week? It's not a sales pitch; I just want to understand the workflow, and I'll share the summary. Happy to keep it off-record or under NDA if you'd prefer.

**Short version for Discord or Slack (very informal):**
> hey [name], quick q since you're at [Company/infra team]: when your power folks submit load models to the utility, do they hit friction on the PSCAD/EMT side? doing research on that bottleneck. not selling anything. would love a 15-min chat or even just a pointer to the right person 🙏

### Template 4: Warm referral or alumni introduction

**To the mutual contact (make it easy to forward):**
> Hi [Mutual Contact],
>
> Hope [personal note]. I'm doing research on how data center power teams handle grid-connection bottlenecks, specifically EMT model validation and the new ERCOT/NERC ride-through rules. I noticed you're connected with **[Target Name]** at [Company].
>
> Would you be comfortable introducing us? Here's a blurb you can forward as-is:
>
> > *"[Your Name] ([School / Former Company]) is researching how power engineers at [neoclouds / data center developers] get through EMT modeling and interconnection delays. They're looking for a 15-min technical conversation: no sales, findings shared back, NDA available."*
>
> No pressure at all if it's not a fit. Thank you!

**Direct alumni note (shared school or former employer):**
> Hi [First Name], fellow [MIT / School] [Course/Year] here (or: ex-[Shared Company]). I saw you're leading [power/grid] work at [Company]. I'm researching EMT model and ride-through bottlenecks for data center interconnections, and your [Hook] made me think you'd have a sharp view. Would you have 15 min in the next couple of weeks? Happy to share what I'm learning from other operators.

---

## 5. Execution playbook

| Week | Action | Target volume |
|---|---|---|
| 1 | Join LinkedIn groups 30, 31, 33 and 34; join the PSCAD Forum, Eng-Tips, r/PowerEngineering, Work on Climate and DER Task Force. Subscribe to ERCOT LLWG and OCP Power Distribution lists. | n/a |
| 1 | Download the ERCOT LLWG and PJM CIFP comment letters and the ESIG LLTF report. Pull named engineers into a CRM. | 50–100 names |
| 2 | Post a Template 3 discussion thread on r/PowerEngineering and in Work on Climate #grid. | 2 posts |
| 2–4 | Send Template 1 to the neocloud boolean list (§3B) and Template 2 to the developer list. | 20 per day, personalized |
| 2–4 | Send Template 4 requests for every 2nd-degree connection with a mutual contact. | 10 per week |
| 3+ | Attend an IEEE PES chapter, a 7x24 chapter or an ESIG workshop session. Follow up within 48 hours and reference the session. | 1–2 events |

**Benchmarks to track:**
- LinkedIn connection acceptance above 35%.
- Reply rate above 15% for personalized messages and above 30% for warm intros.
- Booked-call rate above 8% of messages sent.
- A/B test the subject line: regulation-led ("NOGRR 282") versus pain-led ("model review time").

**Approaches to avoid:**
- Pitching in working-group meetings.
- Mass-tagging people on Reddit.
- Attaching decks.
- Asking for "feedback on our product" in the first message.
- Claiming CEII or confidential utility data. Ask only about the engineer's own workflow.

---

## Sources

- ERCOT, *Dynamic Modeling of AI Data Center Load in PSCAD* (2026-04-24): https://www.ercot.com/files/docs/2026/04/24/Dynamic-Modeling-of-AI-Data-Center-Load-in-PSCAD-Manual.pdf · https://www.ercot.com/files/docs/2026/04/24/AIDataCenter_PSCAD-Manual.pdf · Zenodo: https://zenodo.org/records/21004140
- ERCOT, LEL Ride-Through Requirements NOGRR 282 / NPRR 1308: https://www.ercot.com/files/docs/2026/03/02/04_LEL-RT-Requirements_SPWG_Feb2026.pdf · NPRR1308: https://www.ercot.com/mktrules/issues/NPRR1308
- mgrid.org, *Texas Orders ERCOT Data Centers to Ride Through Grid Faults*: https://mgrid.org/2026/07/21/texas-regulators-order-data-centers-to-ride-through-grid-faults-with-compliance-costs-up-to-1-million-per-mw/
- EPE, *NOGRR 282*: https://epeconsulting.com/epe-intelligence/news/preparing-for-nogrr-282-ercot-s-next-phase-of-large-load-requirements · on.energy guide: https://www.on.energy/regulatory/nogrr282-ercot-ride-through-rule-guide · Keentel: https://keentelengineering.com/ercot-ride-through-requirements
- ZEG, ERCOT large load 2026 and LLWG summaries: https://www.zeroemissiongrid.com/insights-press-zeg-blog/ercot-large-load-interconnection-2026/ · https://www.zeroemissiongrid.com/iso-rto-meeting-summaries/ercot-llwg-05-21/ · https://www.zeroemissiongrid.com/iso-rto-meeting-summaries/ercot-tac-04-26/
- ERCOT LLWG: https://www.ercot.com/committees/tac/llwg
- NERC Level 3 alert: https://www.dwt.com/blogs/energy--environmental-law-blog/2026/05/nerc-level-3-alert-large-loads-data-centers · https://www.utilitydive.com/news/data-center-load-disruptions-nerc-alert-recommendations/818036/ · https://www.morganlewis.com/blogs/datacenterbytes/2026/05/nerc-alert-signals-growing-reliability-focus-on-data-centers · https://www.carbon-direct.com/insights/nerc-level-3-alert-data-center-loads
- NERC LLTF/LLWG: https://www.publicpower.org/periodical/article/nercs-large-loads-task-force-releases-first-white-paper · https://www.nerc.com/globalassets/who-we-are/standing-committees/rstc/3_doc_white-paper-characteristics-and-risks-of-emerging-large-loads.pdf · https://www.whitecase.com/insight-alert/nerc-tees-plan-assess-grid-risks-associated-data-centers
- ESIG Large Loads Task Force: https://www.esig.energy/working-groups/large-loads/ · https://www.esig.energy/wp-content/uploads/2026/04/ESIG-Large-Loads-Modeling-for-Dynamic-Studies-report-2026d.pdf · https://www.esig.energy/event/large-load-modeling-testing-and-interconnection-requirements-workshop/
- EPRI DCFlex: https://www.latitudemedia.com/news/epri-takes-its-data-center-flexibility-project-global/ · https://www.utilitydive.com/news/epri-launches-data-center-flexibility-initiative-with-NVIDIA-google-meta/731490/
- IEEE PES Substations WG T1: https://cmte.ieee.org/pes-substations/sct0/wgt1/ · IEEE DataPort EMT data center models: https://ieee-dataport.org/open-access/electromagnetic-transient-emt-modeling-data-centers
- CIGRE: https://www.cigre.org/article/home/cigre-active-working-groups--call-for-experts · https://cse.cigre.org/cse-n041/integration-of-large-power-electronics-interfaced-loads-in-power-systems-challenges-and-opportunities-part-i-technical-foundations.html
- OCP: https://www.opencompute.org/community/power-distribution · https://www.opencompute.org/wiki/Data_Center_Facility/Power · https://ocp-all.groups.io/
- PJM CIFP: https://www.pjm.com/-/media/DotCom/about-pjm/who-we-are/public-disclosures/2026/20260116-pjm-board-letter-re-results-of-the-cifp-process-large-load-additions.pdf · https://rmi.org/resources/unpacking-the-pjm-cifp-decision-what-pjm-states-can-do-to-ensure-affordable-reliable-electricity-during-the-data-center-boom/
- UK/NESO: https://datacentreaxis.com/uk/intel/uk-grid-connection-queue/ · https://cms.law/en/gbr/legal-updates/spring-2026-update-on-further-demand-connection-reform-for-data-centres-and-other-large-energy-users · https://greensoftware.foundation/policy/research/data-centres-ofgem-response/
- PSCAD (30,000+ users): https://keentelengineering.com/white-paper-pscad-power-system-design · https://www.pscad.com/ · SPP EMT requirements: https://opsportal.spp.org/documents/studies/SPP%20EMT_Model_Requirements_R1.pdf · Olsson job posting: https://www.remoterocketship.com/us/company/olssonassociates/jobs/power-systems-engineer-pscad-data-centers-united-states-remote/
- Reddit stats: https://reddapi.dev/subreddits/datacenter/insights · https://reddapi.dev/subreddits/electricalengineering/insights · https://thehiveindex.com/communities/r-electricalengineering/
- Eng-Tips: https://www.eng-tips.com/ · https://en.wikipedia.org/wiki/Eng-Tips_Forums · openmod: https://forum.openmod.org/ · PyPSA: https://pypsa.org/
- ServeTheHome: https://forums.servethehome.com/index.php · GPU MODE: https://discord.com/invite/g5gAD2vFcm · Data Center Dude: https://datacenterdude.com/community/
- Slack communities: https://workonclimate.org/our-community/ · https://www.canarymedia.com/articles/fun-stuff/here-are-some-of-climatetechs-best-online-communities · https://mcj.vc/members · https://www.canarymedia.com/articles/fun-stuff/friday-social-meet-the-task-force-fighting-for-distributed-energy-resources · https://climateaction.tech/
- LinkedIn groups: https://www.datacenterknowledge.com/operations-and-management/data-center-resources-top-data-center-industry-groups · https://www.dcsmi.com/data-center-sales-and-marketing-blog/refine-your-data-center-strategy-on-these-linkedin-groups
- Associations: https://datacentremagazine.com/articles/top-10-data-centre-associations-and-their-key-drivers · https://www.linkedin.com/company/7x24-exchange-international · https://afcom.com/page/Midlantic
