<template>
    <div class="row" style="margin-top:40px;">
        <div class="column" style="width:4%"></div>
        <div class="column" style="width:36%"><p><b>[Beta-version]</b> {{updates}}</p></div>
        <div class="column" style="width:32%"></div>
        <div class="column" style="width:28%"><p><b style="font-size:17px;">©</b> Kilian, Advaith, Arun, Moritz, Vasileios, Mahir</p><a class="txt" style="color:black;margin-top:-25px;" href="https://www.maastrichtuniversity.nl/p70070981">Supervisor: Dr. Rohan Nanda</a></div>
    </div>
</template>

<script>
export default {
    props: {
        updates: {
            type: String,
            default: 'Default text'
        }
    }
}
</script>

<style>

</style>